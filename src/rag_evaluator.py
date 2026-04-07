import os
import json
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class RAGEvaluator:
    def __init__(self, llm_model: str = "qwen-turbo", embed_model: str = "text-embedding-v3"):
        # 🔑 强制加载环境变量
        load_dotenv()
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("❌ 请检查 .env 文件，确保包含有效的 DASHSCOPE_API_KEY")

        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # 🔑 修复 401：LangChain 新版本必须使用 openai_api_key / openai_api_base
        self.llm = ChatOpenAI(
            model=llm_model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.0,
            max_retries=2
        )
        self.embeddings = OpenAIEmbeddings(
            model=embed_model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_retries=2
        )

    def run_evaluation(self, eval_data_path: str) -> dict:
        if not os.path.exists(eval_data_path):
            raise FileNotFoundError(f"❌ 评估数据文件不存在: {eval_data_path}")

        with open(eval_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        for col in ['question', 'answer', 'contexts', 'ground_truth']:
            if col not in df.columns:
                df[col] = [""] * len(df)

        has_ground_truth = any(str(gt).strip() not in ["", "N/A"] for gt in df['ground_truth'])
        metrics = [faithfulness, answer_relevancy, context_precision]
        if has_ground_truth:
            metrics.append(context_recall)
            print("✅ 检测到 ground_truth，将评估 Context Recall")
        else:
            print("⚠️ 未提供 ground_truth，跳过 Context Recall 评估")

        dataset = Dataset.from_pandas(df)
        print("🔄 正在运行 RAGAS 评估 (约需 2-5 分钟，请勿中断)...")
        result = evaluate(dataset, metrics=metrics, llm=self.llm, embeddings=self.embeddings)

        print("\n📊 RAG 质量评估结果:")
        print(result)

        # 🔑 修复 list.items 报错：兼容 ragas 0.1.x ~ 0.2.x 返回结构
        try:
            if hasattr(result, 'to_pandas'):
                metrics_dict = result.to_pandas().mean(numeric_only=True).to_dict()
            elif hasattr(result, 'scores'):
                if isinstance(result.scores, dict):
                    metrics_dict = result.scores
                else:
                    metrics_dict = {m: sum(s.get(m, 0) for s in result.scores) / len(result.scores) for m in metrics}
            else:
                metrics_dict = dict(result)
        except Exception as e:
            print(f"⚠️ 指标解析异常: {e}，尝试降级处理")
            metrics_dict = {}

        report_path = eval_data_path.replace("eval_data", "rag_eval_report")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({"metrics": {k: round(v, 4) for k, v in metrics_dict.items()}}, f, indent=2)
        print(f"💾 评估报告已保存至: {report_path}")
        return metrics_dict
