import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.retrieval import VectorRetriever, HybridRetriever
from src.api_requests import APIProcessor
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures
import time
import faiss
import jieba
import pickle
from rank_bm25 import BM25Okapi

import tempfile
import shutil
import os
import numpy as np

class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = './vector_dbs',
        documents_dir: Union[str, Path] = './documents',
        questions_file_path: Optional[Union[str, Path]] = None,
        new_challenge_pipeline: bool = False,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,  # 是否启用父文档检索
        llm_reranking: bool = False,              # 是否启用LLM重排
        llm_reranking_sample_size: int = 5,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "dashscope", # openai
        answering_model: str = "qwen-turbo-latest", # gpt-4o-2024-08-06
        full_context: bool = False,
        use_bm25_db: bool = False,
        bm25_db_dir: Optional[Path] = None,
    ):
        # 初始化问题处理器，配置检索、模型、并发等参数
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None
        
        self.new_challenge_pipeline = new_challenge_pipeline
        self.return_parent_pages = parent_document_retrieval
        self.llm_reranking = llm_reranking
        self.llm_reranking_sample_size = llm_reranking_sample_size
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context
        self.use_bm25_db = use_bm25_db
        self.bm25_db_dir = bm25_db_dir
        self.bm25_indices = {}  # 缓存 {sha1: BM25Okapi}
        # 加载 BGE-M3 模型用于查询向量化
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.embedder.max_seq_length = 512

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()
        self._print_lock = threading.Lock()
        self.eval_data = []  # 存储用于 RAGAS 评估的数据
        self._rate_limit_delay = 0.15  # 约 6 QPS，为 500 QPM 留足安全余量
        self.companies_df = None  # 新增：预置属性

    def _print_retrieval_context(self, question: str, results: list, company: str = ""):
        """线程安全的检索内容打印"""
        with self._print_lock:
            print(f"\n{'='*70}")
            print(f"🔍 [检索日志] 公司: {company or 'N/A'} | 问题: {question[:60]}...")
            for i, doc in enumerate(results[:3]):  # 仅打印 Top 3 避免刷屏
                score = doc.get('combined_score', doc.get('relevance_score', 0.0))
                page = doc.get('page', '?')
                content_preview = doc.get('text', '')[:150].replace('\n', ' ').strip()
                print(f"  📄 [{i+1}] Page {page} | 综合得分: {score:.4f}")
                print(f"     📝 {content_preview}...")
            print(f"{'='*70}\n")

    def _load_questions(self, questions_file_path: Optional[Union[str, Path]]) -> List[Dict[str, str]]:
        # 加载问题文件，返回问题列表
        if questions_file_path is None:
            return []
        with open(questions_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """将检索结果格式化为RAG上下文字符串"""
        if not retrieval_results:
            return ""
        
        context_parts = []
        for result in retrieval_results:
            page_number = result['page']
            text = result['text']
            context_parts.append(f'Text retrieved from page {page_number}: \n"""\n{text}\n"""')
            
        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, pages_list: list, company_name: str) -> list:
        # 根据公司名和页码列表，提取引用信息
        self._load_subset_df()  # 安全加载，不会重复打印

        matching_rows = self.companies_df[self.companies_df['company_name'] == company_name]
        company_sha1 = matching_rows.iloc[0]['sha1'] if not matching_rows.empty else ""

        refs = []
        for page in pages_list:
            refs.append({"pdf_sha1": company_sha1, "page_index": page})
        return refs

    def _validate_page_references(self, claimed_pages: list, retrieval_results: list, min_pages: int = 2, max_pages: int = 8) -> list:
        """
        校验LLM答案中引用的页码是否真实存在于检索结果中。
        若不足最小页数，则补充检索结果中的top页。
        """
        if claimed_pages is None:
            claimed_pages = []
        
        retrieved_pages = [result['page'] for result in retrieval_results]
        
        validated_pages = [page for page in claimed_pages if page in retrieved_pages]
        
        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(f"Warning: Removed {len(removed_pages)} hallucinated page references: {removed_pages}")
        
        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)
            
            for result in retrieval_results:
                page = result['page']
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)
                    
                    if len(validated_pages) >= min_pages:
                        break
        
        if len(validated_pages) > max_pages:
            print(f"Trimming references from {len(validated_pages)} to {max_pages} pages")
            validated_pages = validated_pages[:max_pages]
        
        return validated_pages

    def get_answer_for_company(self, company_name: str, question: str, schema: str) -> dict:
        # 针对单个公司，检索上下文并调用LLM生成答案
        t0 = time.time()
        if self.llm_reranking:
            # 🔑 传入 api_provider 和 answering_model
            retriever = HybridRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir,
                bm25_db_dir=self.bm25_db_dir,  # 新增
                rerank_provider=self.api_provider,
                rerank_model=self.answering_model
            )
        else:
            retriever = VectorRetriever(
                vector_db_dir=self.vector_db_dir,
                documents_dir=self.documents_dir
            )
        t1 = time.time()
        print(f"[计时] [get_answer_for_company] 检索器初始化耗时: {t1-t0:.2f} 秒")
        if self.full_context:
            retrieval_results = retriever.retrieve_all(company_name)
        else:           
            t2 = time.time()
            retrieval_results = retriever.retrieve_by_company_name(
                company_name=company_name,
                query=question,
                retrieval_mode="hybrid",  # 选择混合模式
                vector_top_k=self.llm_reranking_sample_size,
                bm25_top_k=self.llm_reranking_sample_size,
                top_n=self.top_n_retrieval,
                debug_print=True  # 开启调试打印
            )
            t3 = time.time()
            print(f"[计时] [get_answer_for_company] 检索耗时: {t3-t2:.2f} 秒")
            self._print_retrieval_context(question, retrieval_results, company_name)

            if not retrieval_results:
                raise ValueError(f"No relevant context found for company: {company_name}")

            rag_context = self._format_retrieval_results(retrieval_results)
        if not retrieval_results:
            raise ValueError("No relevant context found")
        t4 = time.time()
        rag_context = self._format_retrieval_results(retrieval_results)
        t5 = time.time()
        print(f"[计时] [get_answer_for_company] 构建rag_context耗时: {t5-t4:.2f} 秒")
        # 🔑 限流保护：严格遵循 500 QPM
        time.sleep(self._rate_limit_delay)
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model
        )
        t6 = time.time()
        print(f"[计时] [get_answer_for_company] LLM调用耗时: {t6-t5:.2f} 秒")

        self.response_data = self.openai_processor.response_data
        if self.new_challenge_pipeline:
            pages = answer_dict.get("relevant_pages", [])
            validated_pages = self._validate_page_references(pages, retrieval_results)
            answer_dict["relevant_pages"] = validated_pages
            answer_dict["references"] = self._extract_references(validated_pages, company_name)
        print(f"[计时] [get_answer_for_company] 总耗时: {t6-t0:.2f} 秒")
        # 🔑 采集评估数据
        self.eval_data.append({
            "question": question,
            "answer": answer_dict.get("final_answer", ""),
            "contexts": [doc.get("text", "") for doc in retrieval_results[:5]],
            "ground_truth": "N/A"  # 若 questions.json 含答案可替换
        })
        return answer_dict

    # 新增：线程安全的子集加载方法
    def _load_subset_df(self):
        """线程安全地加载 subset.csv，仅执行一次"""
        if self.companies_df is not None:
            return
        with self._lock:  # 使用已定义的 threading.Lock
            if self.companies_df is None:
                if self.subset_path is None:
                    raise ValueError("subset_path must be provided.")
                self.companies_df = pd.read_csv(self.subset_path, encoding='gbk')
                print("✅ subset.csv 已成功加载至内存（仅一次）")


    def _extract_companies_from_subset(self, question_text: str) -> list[str]:
        """从问题文本中提取公司名，匹配subset文件中的公司"""
        self._load_subset_df()  # 安全加载，不会重复打印

        found_companies = []
        company_names = sorted(self.companies_df['company_name'].unique(), key=len, reverse=True)

        for company in company_names:
            if company in question_text:
                found_companies.append(company)
                question_text = question_text.replace(company, '')

        return found_companies

    def process_question(self, question: str, schema: str):
        # 处理单个问题，支持多公司比较
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)
        
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")
        
        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            answer_dict = self.get_answer_for_company(company_name=company_name, question=question, schema=schema)
            return answer_dict
        else:
            return self.process_comparative_question(question, extracted_companies, schema)
    
    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """创建答案详情的引用ID，并存储详细内容"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict['step_by_step_analysis'],
                "reasoning_summary": answer_dict['reasoning_summary'],
                "relevant_pages": answer_dict['relevant_pages'],
                "response_data": self.response_data,
                "self": ref_id
            }
        return ref_id

    def _calculate_statistics(self, processed_questions: List[dict], print_stats: bool = False) -> dict:
        """统计处理结果，包括总数、错误数、N/A数、成功数"""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(1 for q in processed_questions if (q.get("value") if "value" in q else q.get("answer")) == "N/A")
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n")
        
        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count
        }

    def process_questions_list(self, questions_list: List[dict], output_path: str = None, submission_file: bool = False, pipeline_details: str = "") -> dict:
        # 批量处理问题列表，支持并行与断点保存，返回处理结果和统计信息
        total_questions = len(questions_list)
        # 给每个问题加索引，便于后续答案详情定位
        questions_with_index = [{**q, "_question_index": i} for i, q in enumerate(questions_list)]
        self.answer_details = [None] * total_questions  # 预分配答案详情列表
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            # 单线程顺序处理
            for question_data in tqdm(questions_with_index, desc="Processing questions"):
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(processed_questions, output_path, submission_file=submission_file, pipeline_details=pipeline_details)
        else:
            # 多线程并行处理
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_threads) as executor:
                        # executor.map 保证结果顺序与输入一致
                        batch_results = list(executor.map(self._process_single_question, batch))
                    processed_questions.extend(batch_results)
                    
                    if output_path:
                        self._save_progress(processed_questions, output_path, submission_file=submission_file, pipeline_details=pipeline_details)
                    pbar.update(len(batch_results))
        
        statistics = self._calculate_statistics(processed_questions, print_stats = True)
        if output_path:
            eval_path = Path(output_path).parent / f"eval_data_{Path(output_path).stem}.json"
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(self.eval_data, f, ensure_ascii=False, indent=2)
            print(f"✅ RAGAS 评估数据已保存至: {eval_path}")
        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)
        
        if self.new_challenge_pipeline:
            question_text = question_data.get("text")
            schema = question_data.get("kind")
        else:
            question_text = question_data.get("question")
            schema = question_data.get("schema")
        try:
            answer_dict = self.process_question(question_text, schema)
            
            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref({
                    "step_by_step_analysis": None,
                    "reasoning_summary": None,
                    "relevant_pages": None
                }, question_index)
                if self.new_challenge_pipeline:
                    return {
                        "question_text": question_text,
                        "kind": schema,
                        "value": None,
                        "references": [],
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref}
                    }
                else:
                    return {
                        "question": question_text,
                        "schema": schema,
                        "answer": None,
                        "error": answer_dict["error"],
                        "answer_details": {"$ref": detail_ref},
                    }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            if self.new_challenge_pipeline:
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": answer_dict.get("final_answer"),
                    "references": answer_dict.get("references", []),
                    "answer_details": {"$ref": detail_ref}
                }
            else:
                return {
                    "question": question_text,
                    "schema": schema,
                    "answer": answer_dict.get("final_answer"),
                    "answer_details": {"$ref": detail_ref},
                }
        except Exception as err:
            return self._handle_processing_error(question_text, schema, err, question_index)

    def _handle_processing_error(self, question_text: str, schema: str, err: Exception, question_index: int) -> dict:
        """
        处理问题处理过程中的异常。
        记录错误详情并返回包含错误信息的字典。
        """
        import traceback
        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {
            "error_traceback": tb,
            "self": error_ref
        }
        
        with self._lock:
            self.answer_details[question_index] = error_detail
        
        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")
        
        if self.new_challenge_pipeline:
            return {
                "question_text": question_text,
                "kind": schema,
                "value": None,
                "references": [],
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref}
            }
        else:
            return {
                "question": question_text,
                "schema": schema,
                "answer": None,
                "error": f"{type(err).__name__}: {error_message}",
                "answer_details": {"$ref": error_ref},
            }

    def _post_process_submission_answers(self, processed_questions: List[dict]) -> List[dict]:
        """
        提交格式后处理：
        1. 页码从1-based转为0-based
        2. N/A答案清空引用
        3. 格式化为比赛提交schema
        4. 包含step_by_step_analysis
        """
        submission_answers = []
        
        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = "N/A" if "error" in q else (q.get("value") if "value" in q else q.get("answer"))
            references = q.get("references", [])
            
            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith("#/answer_details/"):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if 0 <= index < len(self.answer_details) and self.answer_details[index]:
                        step_by_step_analysis = self.answer_details[index].get("step_by_step_analysis")
                except (ValueError, IndexError):
                    pass
            
            # Clear references if value is N/A
            if value == "N/A":
                references = []
            else:
                # Convert page indices from one-based to zero-based (competition requires 0-based page indices, but for debugging it is easier to use 1-based)
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": ref["page_index"] - 1
                    }
                    for ref in references
                ]
            
            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }
            
            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis
            
            submission_answers.append(submission_answer)
        
        return submission_answers

    def _save_progress(self, processed_questions: List[dict], output_path: Optional[str], submission_file: bool = False, pipeline_details: str = ""):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)
            
            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(output_file.stem + "_debug" + output_file.suffix)
            with open(debug_file, 'w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
            
            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(processed_questions)
                submission = {
                    "answers": submission_answers,
                    "details": pipeline_details
                }
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(self, output_path: str = 'questions_with_answers.json', submission_file: bool = False, pipeline_details: str = ""):
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            pipeline_details=pipeline_details
        )
        return result

    def process_comparative_question(self, question: str, companies: List[str], schema: str) -> dict:
        """
        处理多公司比较类问题：
        1. 先将比较问题重写为单公司问题
        2. 并行处理每个公司
        3. 汇总结果并生成最终比较答案
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question,
            companies=companies
        )
        
        individual_answers = {}
        aggregated_references = []
        
        # Step 2: Process each individual question in parallel
        def process_company_question(company: str) -> tuple[str, dict]:
            """Helper function to process one company's question and return (company, answer)"""
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(f"Could not generate sub-question for company: {company}")
            
            answer_dict = self.get_answer_for_company(
                company_name=company, 
                question=sub_question, 
                schema="number"
            )
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company 
                for company in companies
            }
            
            for future in concurrent.futures.as_completed(future_to_company):
                try:
                    company, answer_dict = future.result()
                    individual_answers[company] = answer_dict
                    
                    company_references = answer_dict.get("references", [])
                    aggregated_references.extend(company_references)
                except Exception as e:
                    company = future_to_company[future]
                    print(f"Error processing company {company}: {str(e)}")
                    raise
        
        # Remove duplicate references
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())
        
        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model
        )
        self.response_data = self.openai_processor.response_data
        
        comparative_answer["references"] = aggregated_references
        return comparative_answer

    def process_single_question(self, question: str, kind: str = "string"):
        """
        单条问题推理，返回结构化答案。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        t0 = time.time()
        print("[计时] [单问] 开始公司名抽取 ...")
        # 公司名抽取
        if self.new_challenge_pipeline:
            extracted_companies = self._extract_companies_from_subset(question)
        else:
            extracted_companies = re.findall(r'"([^"]*)"', question)
        t1 = time.time()
        print(f"[计时] [单问] 公司名抽取耗时: {t1-t0:.2f} 秒")
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")
        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            print("[计时] [单问] 开始检索与LLM推理 ...")
            t2 = time.time()
            answer_dict = self.get_answer_for_company(company_name=company_name, question=question, schema=kind)
            t3 = time.time()
            print(f"[计时] [单问] 检索+LLM推理耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict
        else:
            print("[计时] [单问] 开始多公司比较 ...")
            t2 = time.time()
            answer_dict = self.process_comparative_question(question, extracted_companies, kind)
            t3 = time.time()
            print(f"[计时] [单问] 多公司比较耗时: {t3-t2:.2f} 秒")
            print(f"[计时] [单问] 总耗时: {t3-t0:.2f} 秒")
            return answer_dict

    def _tokenize_query(self, query: str) -> List[str]:
        """与 BM25Ingestor 保持一致的 jieba 分词"""
        words = jieba.lcut(query)
        return [w for w in words if w.strip()]

    def _load_bm25_index(self, sha1: str) -> Optional[BM25Okapi]:
        if sha1 in self.bm25_indices:
            return self.bm25_indices[sha1]
        pkl_path = self.bm25_db_dir / f"{sha1}.pkl"
        if not pkl_path.exists():
            return None
        with open(pkl_path, 'rb') as f:
            index = pickle.load(f)
        self.bm25_indices[sha1] = index
        return index

    def _bm25_search(self, query: str, sha1: str, top_k: int = 10) -> List[Dict]:
        index = self._load_bm25_index(sha1)
        if index is None:
            return []
        tokens = self._tokenize_query(query)
        scores = index.get_scores(tokens)
        # 获取 top_k 索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "chunk_index": idx,
                "score": scores[idx],
                "retriever": "bm25"
            })
        return results

    def _reciprocal_rank_fusion(self, results_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """RRF 融合多个检索结果列表，每个元素需包含 'chunk_index' 字段"""
        scores = {}
        for results in results_lists:
            for rank, item in enumerate(results, start=1):
                idx = item['chunk_index']
                scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"chunk_index": idx, "fusion_score": score} for idx, score in sorted_items]

    def _safe_load_faiss(self, faiss_path: Path) -> Optional[faiss.Index]:
        """
        安全加载 FAISS 索引，彻底绕过 Windows 中文路径限制
        原理：复制到纯 ASCII 临时路径 -> 读取 -> 自动清理
        """
        if not faiss_path.exists():
            return None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_file = os.path.join(tmpdir, "idx.faiss")
                shutil.copy2(str(faiss_path), tmp_file)
                return faiss.read_index(tmp_file)
        except Exception as e:
            print(f"[ERROR] 无法读取 FAISS 索引 {faiss_path.name}: {e}")
            return None

    def _vector_search(self, query: str, sha1: str, top_k: int = 10) -> List[Dict]:
        """安全向量检索（已绕过中文路径 Bug）"""
        faiss_path = self.vector_db_dir / f"{sha1}.faiss"
        index = self._safe_load_faiss(faiss_path)
        if index is None:
            return []

        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0].reshape(1, -1).astype(np.float32)
        scores, indices = index.search(query_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "chunk_index": int(idx),
                    "score": float(scores[0][i]),
                    "retriever": "vector"
                })
        return results

    def _retrieve_chunks(self, query: str, sha1: str, top_n: int = 10) -> List[Dict]:
        """
        混合检索入口（已修复 self.chunks 未定义问题）
        自动从 JSON 加载 chunks，支持向量+BM25 RRF 融合
        """
        # 1. 动态加载 chunks（解决 AttributeError）
        json_path = self.documents_dir / f"{sha1}_chunks.json"
        if not json_path.exists():
            print(f"[WARNING] 未找到 chunks 文件: {json_path.name}")
            return []

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chunks = data.get("content", {}).get("chunks", [])
        if not chunks: return []

        # 2. 执行检索
        vector_results = self._vector_search(query, sha1, top_k=self.top_n_retrieval)
        all_results = [vector_results]

        if self.use_bm25_db:
            bm25_results = self._bm25_search(query, sha1, top_k=self.top_n_retrieval)
            all_results.append(bm25_results)

        # 3. RRF 融合
        fused = self._reciprocal_rank_fusion(all_results, k=60)

        # 4. 组装最终 chunk 数据
        final_chunks = []
        for item in fused[:top_n]:
            idx = item['chunk_index']
            if 0 <= idx < len(chunks):
                chunk_data = chunks[idx].copy()
                chunk_data["fusion_score"] = item['fusion_score']
                final_chunks.append(chunk_data)
        return final_chunks
