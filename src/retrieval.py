import json
import logging
from typing import List, Tuple, Dict, Union
from rank_bm25 import BM25Okapi
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from src.reranking import LLMReranker
import hashlib
import pandas as pd
import time
import tempfile
import shutil
import faiss
from pathlib import Path
import threading
import torch
from sentence_transformers import SentenceTransformer

_log = logging.getLogger(__name__)

# 🔑 添加此函数，彻底解决 Windows 中文路径 + NameError 问题
def _safe_read_faiss(faiss_path: Path) -> faiss.Index:
    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS 索引不存在: {faiss_path}")
    # 复制到临时纯 ASCII 路径供 C++ 底层读取
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = os.path.join(tmpdir, "tmp_idx.faiss")
        shutil.copy2(str(faiss_path), tmp_file)
        return faiss.read_index(tmp_file)


class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        # 初始化BM25检索器，指定BM25索引和文档目录
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir

    def retrieve_by_company_name(self, company_name: str, query: str, top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break
        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")

        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1']}.pkl"
        with open(bm25_path, 'rb') as f:
            bm25_index = pickle.load(f)

        chunks = document["content"].get("chunks", [])
        # 🔑 删除原代码的 pages = document["content"]["pages"]，页码已内嵌在 chunk 中

        # 🔑 核心修复：中文 BM25 必须分词，否则 score 几乎为 0
        import jieba
        tokenized_query = [w for w in jieba.lcut(query) if w.strip()]

        scores = bm25_index.get_scores(tokenized_query)
        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:actual_top_n]

        retrieval_results = []
        seen_pages = set()

        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]

            # 🔑 安全提取页码：兼容 list[int] 或 int，下游引用需要单值
            raw_page = chunk.get("page") or chunk.get("pages", [0])
            page_num = raw_page[0] if isinstance(raw_page, list) else raw_page

            if return_parent_pages:
                if page_num not in seen_pages:
                    seen_pages.add(page_num)
                    # 注：当前 JSON 无独立 page 完整文本，降级返回 chunk 文本
                    result = {"distance": score, "page": page_num, "text": chunk["text"]}
                    retrieval_results.append(result)
            else:
                result = {"distance": score, "page": page_num, "text": chunk["text"]}
                retrieval_results.append(result)

        return retrieval_results

class VectorRetriever:
    _embedder = None
    _load_lock = threading.Lock()  # 🔑 线程安全锁，防止并发重复加载
    def __init__(self, vector_db_dir: Path, documents_dir: Path, embedding_provider: str = "dashscope"):
        # 初始化向量检索器，加载所有向量库和文档
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        # 默认使用 dashscope 作为 embedding provider
        self.embedding_provider = embedding_provider.lower()
        self.llm = self._set_up_llm()

        # 🔑 双重检查锁 + 安全初始化，彻底绕过 meta tensor 报错
        if VectorRetriever._embedder is None:
            with VectorRetriever._load_lock:
                if VectorRetriever._embedder is None:
                    try:
                        # 移除 device='cpu' 参数（触发 meta tensor bug 的元凶）
                        VectorRetriever._embedder = SentenceTransformer(
                            "paraphrase-multilingual-MiniLM-L12-v2"
                        )
                    except Exception as e:
                        print(f"[WARN] 默认加载失败，尝试强制 CPU 降级: {e}")
                        # 降级方案：先加载再移动
                        VectorRetriever._embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

                    # 显式确保在 CPU 运行，避免 Windows 下默认探测异常
                    if not torch.cuda.is_available():
                        VectorRetriever._embedder.to('cpu')
                    VectorRetriever._embedder.eval()

        self.query_embedder = VectorRetriever._embedder

    def _get_embedding(self, query: str):
        """安全获取查询向量"""
        try:
            vec = self.query_embedder.encode([query], normalize_embeddings=True)[0]
            return vec.tolist()
        except Exception as e:
            print(f"[ERROR] ❌ 查询向量化失败: {e}")
            return None

    def _set_up_llm(self):
        # 根据 embedding_provider 初始化对应的 LLM 客户端
        load_dotenv()
        if self.embedding_provider == "openai":
            llm = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=None,
                max_retries=2
            )
            return llm
        elif self.embedding_provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return None  # dashscope 不需要 client 对象
        else:
            raise ValueError(f"不支持的 embedding provider: {self.embedding_provider}")

    def _load_dbs(self):
        # 加载所有向量库和对应文档，建立映射
        all_dbs = []
        all_documents_paths = list(self.documents_dir.glob('*.json'))
        for document_path in all_documents_paths:
            try:
                with open(document_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue
            # 用 metainfo['sha1'] 拼接 faiss 文件名
            sha1 = document.get('metainfo', {}).get('sha1', None)
            if not sha1:
                _log.warning(f"No sha1 found in metainfo for document {document_path.name}")
                continue
            faiss_path = self.vector_db_dir / f"{sha1}.faiss"
            if not faiss_path.exists():
                _log.warning(f"No matching vector DB found for document {document_path.name} (sha1={sha1})")
                continue
            try:
                vector_db = _safe_read_faiss(faiss_path)
                if vector_db is None: raise ValueError(...)
            except Exception as e:
                _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                continue
            report = {
                "name": sha1,
                "vector_db": vector_db,
                "document": document
            }
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        # 计算两个字符串的余弦相似度（通过嵌入）
        llm = VectorRetriever.set_up_llm()
        embeddings = llm.embeddings.create(input=[str1, str2], model="text-embedding-3-large")
        embedding1 = embeddings.data[0].embedding
        embedding2 = embeddings.data[1].embedding
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(self, company_name: str, query: str, llm_reranking_sample_size: int = None,
                                 top_n: int = 3, return_parent_pages: bool = False) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo", {})
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
            elif company_name in metainfo.get("file_name", ""):
                target_report = report
                break
        if target_report is None:
            raise ValueError(f"No report found with '{company_name}' company name.")

        sha1 = target_report["document"]["metainfo"].get("sha1")
        if not sha1:
            raise ValueError(f"No sha1 found in metainfo for company '{company_name}'")

        faiss_path = self.vector_db_dir / f"{sha1}.faiss"
        if not faiss_path.exists():
            raise ValueError(f"No vector DB found for '{company_name}' (sha1: {sha1})")

        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"].get("chunks", [])
        actual_top_n = min(top_n, len(chunks))

        embedding = self._get_embedding(query)
        if embedding is None:
            raise RuntimeError("Query embedding failed.")
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)

        retrieval_results = []
        seen_pages = set()

        for distance, index in zip(distances[0], indices[0]):
            if index == -1: continue  # FAISS 未找到时的占位符
            distance = round(float(distance), 4)
            chunk = chunks[index]

            # 🔑 同步 BM25 的页码提取逻辑，确保下游引用格式统一
            raw_page = chunk.get("page") or chunk.get("pages", [0])
            page_num = raw_page[0] if isinstance(raw_page, list) else raw_page

            if return_parent_pages:
                if page_num not in seen_pages:
                    seen_pages.add(page_num)
                    result = {"distance": distance, "page": page_num, "text": chunk["text"]}
                    retrieval_results.append(result)
            else:
                result = {"distance": distance, "page": page_num, "text": chunk["text"]}
                retrieval_results.append(result)

        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        # 检索公司所有文本块，返回全部内容
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break
        
        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")
        
        document = target_report["document"]
        pages = document["content"]["pages"]
        
        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {
                "distance": 0.5,
                "page": page["page"],
                "text": page["text"]
            }
            all_pages.append(result)
            
        return all_pages


# src/retrieval.py (仅替换 HybridRetriever 类)
class HybridRetriever:
    # ✅ 修复：默认模型改为 qwen-turbo-latest
    def __init__(self, vector_db_dir: Path, documents_dir: Path, bm25_db_dir: Path,
                 rerank_provider: str = "dashscope", rerank_model: str = "qwen-turbo-latest"):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.bm25_db_dir = bm25_db_dir

        # 实例化子检索器
        self.vector_retriever = VectorRetriever(vector_db_dir, documents_dir)
        self.bm25_retriever = BM25Retriever(bm25_db_dir, documents_dir)

        # ✅ 正确传入 model 参数
        self.reranker = LLMReranker(provider=rerank_provider, model=rerank_model)

    def retrieve_by_company_name(self, company_name: str, query: str, retrieval_mode: str = "hybrid",
                                 vector_top_k: int = 10, bm25_top_k: int = 10, top_n: int = 6,
                                 debug_print: bool = False, return_parent_pages: bool = False) -> List[Dict]:
        # (保持你原有的 retrieve_by_company_name 逻辑不变)
        vector_results = []
        bm25_results = []

        if retrieval_mode in ("hybrid", "vector"):
            if debug_print: print("[DEBUG] 开始向量检索...")
            vector_results = self.vector_retriever.retrieve_by_company_name(
                company_name=company_name, query=query, top_n=vector_top_k,
                return_parent_pages=return_parent_pages
            )

        if retrieval_mode in ("hybrid", "bm25"):
            if debug_print: print("[DEBUG] 开始BM25检索...")
            bm25_results = self.bm25_retriever.retrieve_by_company_name(
                company_name=company_name, query=query, top_n=bm25_top_k,
                return_parent_pages=return_parent_pages
            )

        if retrieval_mode == "hybrid":
            fused_results = self._rrf_fusion(vector_results, bm25_results, k=60)
            if debug_print: print(f"[DEBUG] RRF融合后: {len(fused_results)} 条")
        elif retrieval_mode == "vector":
            fused_results = vector_results
        elif retrieval_mode == "bm25":
            fused_results = bm25_results
        else:
            raise ValueError(f"不支持的检索模式: {retrieval_mode}")

        if self.reranker and retrieval_mode == "hybrid":
            if debug_print: print("[DEBUG] 开始LLM重排...")
            fused_results = self.reranker.rerank_documents(query, fused_results)
        else:
            fused_results.sort(key=lambda x: x.get("distance", 0), reverse=True)

        return fused_results[:top_n]

    @staticmethod
    def _rrf_fusion(list1: List[Dict], list2: List[Dict], k: int = 60) -> List[Dict]:
        # (保持你原有的 _rrf_fusion 逻辑不变)
        scores = {}

        def get_key(doc):
            return f"{doc.get('page')}_{doc.get('text', '')[:50]}"

        for rank, item in enumerate(list1, start=1):
            key = get_key(item)
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank)
        for rank, item in enumerate(list2, start=1):
            key = get_key(item)
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank)

        sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result, seen = [], set()
        for key, score in sorted_keys:
            doc = next((d for d in list1 + list2 if get_key(d) == key), None)
            if doc and key not in seen:
                seen.add(key)
                doc_copy = doc.copy()
                doc_copy["rrf_score"] = round(score, 4)
                doc_copy["distance"] = round(score, 4)
                result.append(doc_copy)
        return result
