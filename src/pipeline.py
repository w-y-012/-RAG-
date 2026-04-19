#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
import sys
import os

# 🔧 Windows 异步 SQLite 兼容性修复（必须放在最前面）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ✅ Phoenix 环境变量前置
os.environ["PX_DISABLE_GRPC"] = "true"
os.environ["PHOENIX_PORT"] = "6006"
os.environ["USE_MOCK_LLM"] = "false"

import time, json, logging
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass
import pandas as pd
from pyprojroot import here
from langgraph.graph import StateGraph, START, END

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

from core_security import SecureAgentRuntime, RuntimeConfig, ContextGuard, TraceContext, ToolPermission
from google_api import GoogleSearchTool
from src.ingestion import VectorDBIngestor, BM25Ingestor
from src.text_splitter import TextSplitter
from src.questions_processing import QuestionsProcessor
from src.rag_evaluator import RAGEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("SecurePipeline")


# ================= 配置 =================
@dataclass
class RunConfig:
    parallel_requests: int = 4
    top_n_retrieval: int = 3
    llm_model: str = "qwen-turbo-latest"
    api_provider: str = "dashscope"
    use_bm25: bool = True
    parent_doc: bool = True


max_config = RunConfig()


# ================= LangGraph 状态 =================
class AgentState(TypedDict):
    root_path: Path
    config: RunConfig
    runtime: SecureAgentRuntime
    google_tool: GoogleSearchTool
    questions: List[Dict[str, Any]]
    answers_path: Path
    results: List[Dict]
    trace_id: str

# ================= 图节点 =================
def init_phoenix(state: AgentState) -> dict:
    return state

def get_next_answer_path(root: Path) -> Path:
    base = root / "answers_sec.json"
    if not base.exists(): return base
    i = 1
    while True:
        p = root / f"answers_sec_{i:02d}.json"
        if not p.exists(): return p
        i += 1

async def data_prep_node(state: AgentState) -> dict:
    root = state["root_path"]
    md_dir = root / "debug_data/03_reports_markdown"
    chunk_dir = root / "databases/chunked_reports"
    vec_dir = root / "databases/vector_dbs"
    bm25_dir = root / "databases/bm25_dbs"

    # ✅ 优化：检查是否已存在向量库，避免重复处理
    if vec_dir.exists() and any(vec_dir.glob("*.faiss")):
        logger.info(f"⏩ 检测到现有向量库 {vec_dir}，跳过预处理 (如需重置请删除该目录)")
        # 但仍需确保 chunk 目录存在，否则检索会报错
        if not chunk_dir.exists():
            logger.warning("⚠️ 向量库存在但分块目录缺失，可能检索失败")
        return state

    logger.info("🔄 开始数据预处理 (分块 + 建库)...")
    splitter = TextSplitter()
    # 确保目录存在
    md_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"🔄 开始数据预处理 {root}")

    splitter.split_mineru_jsons(md_dir, chunk_dir, root / "subset.csv")

    vec_ing = VectorDBIngestor()
    await asyncio.to_thread(vec_ing.process_reports, chunk_dir, vec_dir)

    if state["config"].use_bm25:
        bm25_ing = BM25Ingestor()
        await asyncio.to_thread(bm25_ing.process_reports, chunk_dir, bm25_dir)

    logger.info("✅ 数据预处理完成")
    return state


async def load_questions_node(state: AgentState) -> dict:
    q_path = state["root_path"] / "questions.json"
    if not q_path.exists(): q_path = state["root_path"] / "questions.txt"
    questions = []
    if q_path.suffix == ".json" and q_path.exists():
        with open(q_path, "r", encoding="utf-8") as f:   # ✅ 添加编码
            questions = json.load(f)
    elif q_path.exists():
        with open(q_path, "r", encoding="utf-8") as f:   # ✅ 添加编码
            questions = [{"text": l.strip()} for l in f if l.strip()]
    else:
        questions = [{"text": "请分析当前科技股财报中的营收趋势"}]
    return {"questions": questions, "answers_path": state["root_path"] / "answers_sec.json"}


async def secure_processing_node(state: AgentState) -> dict:
    proc = QuestionsProcessor(
        vector_db_dir=state["root_path"] / "databases/vector_dbs",
        documents_dir=state["root_path"] / "databases/chunked_reports",
        questions_file_path=None, new_challenge_pipeline=True,
        subset_path=state["root_path"] / "subset.csv", parent_document_retrieval=state["config"].parent_doc,
        llm_reranking=False, top_n_retrieval=state["config"].top_n_retrieval,
        parallel_requests=state["config"].parallel_requests, api_provider=state["config"].api_provider,
        answering_model=state["config"].llm_model, use_bm25_db=state["config"].use_bm25,
        bm25_db_dir=state["root_path"] / "databases/bm25_dbs",
        runtime=state["runtime"]
    )

    results = []
    for q_data in state["questions"]:
        trace = TraceContext.new()
        q_text = q_data.get("text", q_data.get("question", ""))
        safe_q = ContextGuard.sanitize_retrieval(q_text)

        # 安全策略校验
        await state["runtime"].policy.check("rag_retrieve", ToolPermission.READ, {"q": q_text})
        await state["runtime"].policy.check("llm_generate", ToolPermission.EXECUTE, {"q_len": len(q_text)})

        # ✅ 修复：正确获取 RAG 检索上下文
        rag_context_text = ""
        rag_doc_count = 0
        ans = {"final_answer": "N/A", "references": []}

        try:
            # 调用处理器
            ans = await proc.process_single_question_async(safe_q, kind="string")

            # ✅ 从 processor 内部获取检索到的文档上下文 (关键修复)
            # 方法 1: 从 eval_data 中获取最近一次检索的 contexts
            if hasattr(proc, '_eval_data') and len(proc._eval_data) > 0:
                last_eval = proc._eval_data[-1]
                rag_contexts = last_eval.get("contexts", [])
                if rag_contexts:
                    rag_context_text = "\n".join([str(c) for c in rag_contexts if c and str(c).strip()])
                    rag_doc_count = len(rag_contexts)

            # 方法 2: 如果 eval_data 为空，尝试从答案中提取引用信息
            if not rag_context_text:
                ans_text = str(ans.get("final_answer", ""))
                if ans_text and len(ans_text) > 10:
                    rag_context_text = ans_text
                    rag_doc_count = 1

        except Exception as e:
            logger.error(f"RAG 处理异常：{e}")
            ans = {"final_answer": f"Error: {e}", "references": []}
            rag_context_text = ""
            rag_doc_count = 0

        # ✅ 修复：基于检索到的文档数量和内容长度判断，而非答案长度
        logger.info(f"📊 RAG 检索结果 | 文档数：{rag_doc_count} | 上下文长度：{len(rag_context_text)}")

        if rag_doc_count == 0 or len(rag_context_text) < 50:
            logger.info(f"🌐 RAG 检索不足 -> Google 兜底 | Q:{q_text[:40]}...")
            await state["runtime"].policy.check("google_search", ToolPermission.READ, {"q": q_text})
            # ✅ 修复：num 改为 num_results
            g_res = await state["google_tool"].search(q_text, num_results=3)
            ctx_text = "\n".join([r["snippet"] for r in g_res.get("results", [])])
            source = "google_fallback"
        else:
            logger.info(f"🔍 RAG 检索成功 -> Google 验证 | Q:{q_text[:40]}...")

            # ✅ 关键修复：传入答案而非问题，获取结构化验证结果
            ans_text = str(ans.get("final_answer", ""))
            score, verdict, matched_sources = await state["google_tool"].verify_fact(
                question=q_text,
                rag_answer=ans_text,
                rag_context=rag_context_text
            )

            ctx_text = rag_context_text
            source = "rag_verified" if score >= 0.45 else "rag_low_conf"

            # 注入验证元数据
            ans["verification_score"] = score
            ans["verification_verdict"] = verdict
            ans["matched_sources"] = matched_sources  # 可用于前端展示“已验证信源”

        results.append({
            "question_text": q_text,
            "value": ans.get("final_answer", "N/A"),
            "schema": "string",
            "contexts": [ctx_text],
            "security": {
                "trace": trace,
                "source": source,
                "verified": ans.get("verification_score", 0),
                "rag_doc_count": rag_doc_count  # ✅ 添加检索文档数用于调试
            }
        })

        # 更新 eval_data 用于后续评估
        if hasattr(proc, '_eval_data'):
            # 确保 contexts 字段正确
            for item in proc._eval_data:
                if item.get("question") == q_text:
                    item["contexts"] = [ctx_text]

    with open(state["answers_path"], "w", encoding="utf-8") as f:
        json.dump({"questions": results}, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 本批次处理完成 | 结果数：{len(results)}")
    return {"results": results}

async def evaluation_node(state: AgentState) -> dict:
    eval_dir = state["answers_path"].parent
    clean_path = eval_dir / "clean_eval.json"

    # ✅ 优化：复用旧版健壮的 DataFrame 清洗逻辑
    try:
        df = pd.DataFrame(state["results"])
        if df.empty:
            logger.warning("⚠️ 无结果可评估")
            return state

        # 标准化列名以适配 RAGAS
        if "question_text" in df.columns: df.rename(columns={"question_text": "question"}, inplace=True)
        if "value" in df.columns: df.rename(columns={"value": "answer"}, inplace=True)

        required_cols = {"question", "answer", "contexts"}
        if not required_cols.issubset(df.columns):
            logger.error(f"❌ 评估文件缺少必要字段：{required_cols - set(df.columns)}")
            return state

        # 强制类型对齐，防止 RAGAS Pydantic 校验失败
        df['answer'] = df['answer'].astype(str)
        df['question'] = df['question'].astype(str)
        df['contexts'] = df['contexts'].apply(
            lambda x: [str(c) for c in x] if isinstance(x, list) else [str(x)])

        df.to_json(clean_path, orient="records", force_ascii=False)

        try:
            RAGEvaluator(enable_phoenix=False).run_evaluation(str(clean_path))
        except Exception as e:
            logger.warning(f"评估执行异常：{e}")
    except Exception as e:
        logger.error(f"评估预处理失败：{e}")
    finally:
        clean_path.unlink(missing_ok=True)
    return state

def cleanup_node(state: AgentState) -> dict:
    return state


# ================= 构建 Graph =================
def build_secure_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("init", init_phoenix)
    g.add_node("prep", data_prep_node)
    g.add_node("load_q", load_questions_node)
    g.add_node("proc", secure_processing_node)
    g.add_node("eval", evaluation_node)
    g.add_node("clean", cleanup_node)

    g.add_edge(START, "init")
    g.add_edge("init", "prep")
    g.add_edge("prep", "load_q")
    g.add_edge("load_q", "proc")
    g.add_edge("proc", "eval")
    g.add_edge("eval", "clean")
    g.add_edge("clean", END)
    return g.compile()


if __name__ == "__main__":
    try:
        # ✅ 修复：新版 Phoenix 已移除 timeout/verbose，改用标准参数
        px.launch_app(port=6006)
        print("🌐 Phoenix UI 已启动: http://localhost:6006")
    except TypeError as e:
        # 兼容极旧版本或参数签名变更
        if "timeout" in str(e) or "verbose" in str(e):
            px.launch_app(port=6006)
            print("🌐 Phoenix UI 已启动: http://localhost:6006")
        else:
            raise
    except RuntimeError as e:
        err_msg = str(e).lower()
        if "address already in use" in err_msg or "port" in err_msg:
            print("ℹ️ Phoenix 已在运行中，自动复用现有实例")
        else:
            print(f"⚠️ Phoenix 启动失败 (跳过 UI 服务继续执行): {e}")
    except Exception as e:
        print(f"⚠️ Phoenix 异常 (不影响 Pipeline 运行): {e}")

    # 注册 OpenTelemetry 追踪
    tp = register(project_name="rag-stock-pipeline", endpoint="http://127.0.0.1:6006/v1/traces")
    LangChainInstrumentor().instrument(tracer_provider=tp)

    root = here().parent / "data" / "stock_data"
    # ✅ 修复：实例化配置并传入 Runtime
    # 原代码：runtime = SecureAgentRuntime()
    runtime = SecureAgentRuntime(cfg=RuntimeConfig())
    google = GoogleSearchTool(mock_mode=(os.getenv("USE_MOCK_LLM") == "true"))

    initial: AgentState = {
        "root_path": root,
        "config": max_config,
        "runtime": runtime,
        "google_tool": google,
        "questions": [],
        # ✅ 优化：动态生成文件名，避免覆盖
        "answers_path": get_next_answer_path(root),
        "results": [],
        "trace_id": ""
    }

    print("\n🚀 启动 Secure LangGraph Pipeline...")
    app = build_secure_graph()
    asyncio.run(app.ainvoke(initial))

    print("⏳ 同步追踪并清理资源...")
    time.sleep(2)
    try:
        tp.shutdown()
        px.close_app()
        print("✅ 完成")
    except Exception as e:
        err_msg = str(e)
        if "WinError 32" in err_msg or "PermissionError" in err_msg or "is still in use" in err_msg:
            print("✅ Windows 端口/文件锁已忽略 (Phoenix 进程自动后台运行)")
        else:
            print(f"⚠️ 清理异常: {e}")

