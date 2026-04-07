import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor

# LLMReranker：基于大模型的重排器，支持 OpenAI/DeepSeek 原生结构化输出
class LLMReranker:
    # ✅ 修复1：显式声明 model 参数，默认值改为 qwen-turbo-latest
    def __init__(self, provider: str = "openai", model: str = "qwen-turbo-latest"):
        self.provider = provider.lower()
        self.model = model
        self.llm = self._set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks

    def _set_up_llm(self):
        load_dotenv()
        if self.provider == "openai":
            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
                timeout=30,
                max_retries=2
            )
        elif self.provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return dashscope
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def _call_llm_with_parse(self, messages, response_format):
        """统一结构化调用，支持 OpenAI 原生 parse & 自动降级"""
        import os
        if os.getenv("USE_MOCK_LLM", "false").lower() == "true":
            print("[MOCK] 使用模拟 LLM 响应（重排）")
            # 根据 response_format 类型返回固定结构
            if hasattr(response_format, 'model_fields'):
                if 'relevance_score' in response_format.model_fields:
                    return {"relevance_score": 0.8, "reasoning": "模拟相关性评分"}
                elif 'block_rankings' in response_format.model_fields:
                    # 简单模拟 3 个段落
                    return {
                        "block_rankings": [
                            {"relevance_score": 0.9, "reasoning": "模拟1"},
                            {"relevance_score": 0.7, "reasoning": "模拟2"},
                            {"relevance_score": 0.5, "reasoning": "模拟3"}
                        ]
                    }
            # 兜底
            return {"relevance_score": 0.5, "reasoning": "模拟降级"}
        try:
            if self.provider == "openai":
                resp = self.llm.beta.chat.completions.parse(
                    model=self.model, temperature=0, messages=messages, response_format=response_format
                )
                return resp.choices[0].message.parsed.model_dump()
            elif self.provider == "dashscope":
                # ✅ 修复2：移除硬编码 model="qwen-turbo"，改用动态传入的 self.model
                rsp = self.llm.Generation.call(model=self.model, messages=messages, temperature=0,
                                               result_format='message')
                if not rsp or getattr(rsp, 'status_code', 200) != 200:
                    raise RuntimeError("DashScope API 返回异常")
                content = rsp.output.choices[0].message.content
                clean = content.replace('```json', '').replace('```', '').strip()
                return json.loads(clean)
        except Exception as e:
            print(f"[WARN] LLM重排调用失败，降级使用向量分数: {e}")
            return None

    def get_rank_for_single_block(self, query, retrieved_document):
            # 中文单块重排 Prompt
            user_prompt = f"""请评估以下检索内容与问题的相关性：
    【问题】
    "{query}"

    【检索内容】
    \"\"\"
    {retrieved_document}
    \"\"\"

    请输出JSON格式，包含 relevance_score (0.0~1.0) 和 reasoning (简短理由)。"""
            messages = [
                {"role": "system",
                 "content": "你是专业的金融文档检索重排专家。请严格评估文本与问题的语义相关性，分数越高表示越相关。"},
                {"role": "user", "content": user_prompt},
            ]
            res = self._call_llm_with_parse(messages, self.schema_for_single_block)
            return res if res else {"relevance_score": 0.5, "reasoning": "API降级"}

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
            # 中文批量重排 Prompt
            formatted_blocks = "\n\n---\n\n".join(
                [f'【段落 {i + 1}】\n\n\"\"\"\n{text}\n\"\"\"' for i, text in enumerate(retrieved_documents)])

            user_prompt = f"""请对以下检索段落与问题的相关性进行打分排序：
    【问题】
    "{query}"

    【检索段落】
    {formatted_blocks}

    请严格按顺序返回 {len(retrieved_documents)} 个段落的评分，输出JSON格式。"""
            messages = [
                {"role": "system",
                 "content": "你是专业的金融文档检索重排专家。请依次评估每个段落与问题的语义相关性，输出 relevance_score (0.0~1.0) 和 reasoning。"},
                {"role": "user", "content": user_prompt},
            ]
            res = self._call_llm_with_parse(messages, self.schema_for_multiple_blocks)
            return res if res else {
                "block_rankings": [{"relevance_score": 0.5, "reasoning": "API降级"} for _ in retrieved_documents]}

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight
        all_results = []

        for batch in doc_batches:
            try:
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                block_rankings = rankings.get('block_rankings', [])
                # 补齐 LLM 返回数量不足的情况
                while len(block_rankings) < len(batch):
                    block_rankings.append({"relevance_score": 0.5, "reasoning": "默认补齐"})

                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank.get("relevance_score", 0.5)
                    doc_with_score["combined_score"] = round(
                        llm_weight * doc_with_score["relevance_score"] +
                        vector_weight * doc.get('distance', 0.5), 4
                    )
                    all_results.append(doc_with_score)
            except Exception as e:
                print(f"[ERROR] 重排批次异常，降级为纯向量排序: {e}")
                for doc in batch:
                    doc_copy = doc.copy()
                    doc_copy["relevance_score"] = 0.5
                    doc_copy["combined_score"] = round(vector_weight * doc.get('distance', 0.5), 4)
                    all_results.append(doc_copy)

        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
