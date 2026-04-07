import os
import json
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
from openai import OpenAI
import asyncio
from src.api_request_parallel_processor import process_api_requests_from_file
from openai.lib._parsing import type_to_response_format_param
import tiktoken
import src.prompts as prompts
import requests
from json_repair import repair_json
from pydantic import BaseModel
from copy import deepcopy
from tenacity import retry, stop_after_attempt, wait_fixed
import dashscope


# OpenAI基础处理器，封装了消息发送、结构化输出、计费等逻辑
class BaseOpenaiProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = 'gpt-4o-2024-08-06'
        # self.default_model = 'gpt-4o-mini-2024-07-18',

    def set_up_llm(self):
        # 加载OpenAI API密钥，初始化LLM
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
        )
        return llm

    def send_message(
            self,
            model=None,
            temperature=0.5,
            seed=None,  # For deterministic ouptputs
            system_content='You are a helpful assistant.',
            human_content='Hello!',
            is_structured=False,
            response_format=None
    ):
        # 发送消息到OpenAI，支持结构化/非结构化输出
        if model is None:
            model = self.default_model
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content}
            ]
        }

        # 部分模型不支持temperature
        if "o3-mini" not in model:
            params["temperature"] = temperature

        if not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        elif is_structured:
            params["response_format"] = response_format
            completion = self.llm.beta.chat.completions.parse(**params)

            response = completion.choices[0].message.parsed
            content = response.dict()

        self.response_data = {"model": completion.model, "input_tokens": completion.usage.prompt_tokens,
                              "output_tokens": completion.usage.completion_tokens}
        print(self.response_data)

        return content

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        # 统计字符串的token数
        encoding = tiktoken.get_encoding(encoding_name)
        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)
        return token_count


# IBM API基础处理器，支持余额查询、模型列表、嵌入、消息发送等
class BaseIBMAPIProcessor:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = 'meta-llama/llama-3-3-70b-instruct'

    def check_balance(self):
        """查询当前API余额"""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}

        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None

    def get_available_models(self):
        """获取可用基础模型列表"""
        models_url = f"{self.base_url}/foundation_model_specs"

        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None

    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """获取文本的向量嵌入"""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": model_id
        }

        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None

    def send_message(
            self,
            # model='meta-llama/llama-3-1-8b-instruct',
            model=None,
            temperature=0.5,
            seed=None,  # For deterministic outputs
            system_content='You are a helpful assistant.',
            human_content='Hello!',
            is_structured=False,
            response_format=None,
            max_new_tokens=5000,
            min_new_tokens=1,
            **kwargs
    ):
        # 发送消息到IBM API，支持结构化/非结构化输出
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]

        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs
        }

        payload = {
            "input": input_messages,
            "model_id": model,
            "parameters": parameters
        }

        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {"model": completion.get("model_id"),
                                  "input_tokens": completion.get("results")[0].get("input_token_count"),
                                  "output_tokens": completion.get("results")[0].get("generated_token_count")}
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                    return content

                except Exception as err:
                    print("Error processing structured response, attempting to reparse the response...")
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = response_format.model_validate(reparsed_dict)
                            print("Reparsing successful!")
                            content = validated_data.model_dump()
                            return content

                        except Exception:
                            return reparsed_dict

                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content

            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):

        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            response=response
        )

        reparsed_response = self.send_message(
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )

        return reparsed_response

class APIProcessor:
    def __init__(self, provider: Literal["openai", "ibm", "gemini", "dashscope"] = "dashscope"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "dashscope":
            self.processor = BaseDashscopeProcessor()

    def send_message(
            self,
            model=None,
            temperature=0.5,
            seed=None,
            system_content="You are a helpful assistant.",
            human_content="Hello!",
            is_structured=False,
            response_format=None,
            **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)

        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format
        )
        self.response_data = self.processor.response_data

        # 检查返回的字典是否包含所需的字段，如果不是dashscope则进行兜底
        if not isinstance(answer_dict, dict) or 'step_by_step_analysis' not in answer_dict:
            # 如果是dashscope返回的基本格式，尝试保留其内容
            if isinstance(answer_dict, dict) and 'final_answer' in answer_dict:
                # 这是dashscope处理后的格式，尝试从final_answer中提取结构化信息
                final_answer_content = answer_dict.get("final_answer", "N/A")

                # 如果final_answer是字符串且包含结构化信息，尝试解析
                if isinstance(final_answer_content, str) and final_answer_content.strip().startswith('{'):
                    try:
                        structured_data = json.loads(final_answer_content)
                        answer_dict = structured_data
                    except json.JSONDecodeError:
                        # 如果final_answer不是JSON，保持原有结构
                        answer_dict = {
                            "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
                            "reasoning_summary": answer_dict.get("reasoning_summary", ""),
                            "relevant_pages": answer_dict.get("relevant_pages", []),
                            "final_answer": answer_dict.get("final_answer", "N/A")
                        }
                else:
                    # 否则使用兜底结构
                    answer_dict = {
                        "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
                        "reasoning_summary": answer_dict.get("reasoning_summary", ""),
                        "relevant_pages": answer_dict.get("relevant_pages", []),
                        "final_answer": answer_dict.get("final_answer", "N/A")
                    }
            else:
                # 如果不是预期格式，进行兜底
                answer_dict = {
                    "step_by_step_analysis": "",
                    "reasoning_summary": "",
                    "relevant_pages": [],
                    "final_answer": "N/A"
                }
        return answer_dict

    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        use_schema_prompt = True if self.provider == "ibm" or self.provider == "gemini" else False

        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema
                             if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                             if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                             if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                             if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                             if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        elif schema == "string":
            # 新增：支持开放性文本问题
            system_prompt = (prompts.AnswerWithRAGContextStringPrompt.system_prompt_with_schema
                             if use_schema_prompt else prompts.AnswerWithRAGContextStringPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextStringPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextStringPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str]) -> Dict[str, str]:
        """Use LLM to break down a comparative question into individual questions."""
        answer_dict = self.processor.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies])
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions
        )

        # Convert the answer_dict to the desired format
        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}

        return questions_dict


class AsyncOpenaiProcessor:

    def _get_unique_filepath(self, base_filepath):
        """Helper method to get unique filepath"""
        if not os.path.exists(base_filepath):
            return base_filepath

        base, ext = os.path.splitext(base_filepath)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"

    async def process_structured_ouputs_requests(
            self,
            model="gpt-4o-mini-2024-07-18",
            temperature=0.5,
            seed=None,
            system_content="You are a helpful assistant.",
            queries=None,
            response_format=None,
            requests_filepath='./temp_async_llm_requests.jsonl',
            save_filepath='./temp_async_llm_results.jsonl',
            preserve_requests=False,
            preserve_results=True,
            request_url="https://api.openai.com/v1/chat/completions",
            max_requests_per_minute=3_500,
            max_tokens_per_minute=3_500_000,
            token_encoding_name="o200k_base",
            max_attempts=5,
            logging_level=20,
            progress_callback=None
    ):
        # Create requests for jsonl
        jsonl_requests = []
        for idx, query in enumerate(queries):
            request = {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query},
                ],
                'response_format': type_to_response_format_param(response_format),
                'metadata': {'original_index': idx}
            }
            jsonl_requests.append(request)

        # Get unique filepaths if files already exist
        requests_filepath = self._get_unique_filepath(requests_filepath)
        save_filepath = self._get_unique_filepath(save_filepath)

        # Write requests to JSONL file
        with open(requests_filepath, "w") as f:
            for request in jsonl_requests:
                json_string = json.dumps(request)
                f.write(json_string + "\n")

        # Process API requests
        total_requests = len(jsonl_requests)

        async def monitor_progress():
            last_count = 0
            while True:
                try:
                    with open(save_filepath, 'r') as f:
                        current_count = sum(1 for _ in f)
                        if current_count > last_count:
                            if progress_callback:
                                for _ in range(current_count - last_count):
                                    progress_callback()
                            last_count = current_count
                        if current_count >= total_requests:
                            break
                except FileNotFoundError:
                    pass
                await asyncio.sleep(0.1)

        async def process_with_progress():
            await asyncio.gather(
                process_api_requests_from_file(
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    request_url=request_url,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_requests_per_minute=max_requests_per_minute,
                    max_tokens_per_minute=max_tokens_per_minute,
                    token_encoding_name=token_encoding_name,
                    max_attempts=max_attempts,
                    logging_level=logging_level
                ),
                monitor_progress()
            )

        await process_with_progress()

        with open(save_filepath, "r") as f:
            validated_data_list = []
            results = []
            for line_number, line in enumerate(f, start=1):
                raw_line = line.strip()
                try:
                    result = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_number}: Failed to load JSON from line: {raw_line}")
                    continue

                # Check finish_reason in the API response
                finish_reason = result[1]['choices'][0].get('finish_reason', '')
                if finish_reason != "stop":
                    print(f"[WARNING] Line {line_number}: finish_reason is '{finish_reason}' (expected 'stop').")

                # Safely parse answer; if it fails, leave answer empty and report the error.
                try:
                    answer_content = result[1]['choices'][0]['message']['content']
                    answer_parsed = json.loads(answer_content)
                    answer = response_format(**answer_parsed).model_dump()
                except Exception as e:
                    print(f"[ERROR] Line {line_number}: Failed to parse answer JSON. Error: {e}.")
                    answer = ""

                results.append({
                    'index': result[2],
                    'question': result[0]['messages'],
                    'answer': answer
                })

            # Sort by original index and build final list
            validated_data_list = [
                {'question': r['question'], 'answer': r['answer']}
                for r in sorted(results, key=lambda x: x['index']['original_index'])
            ]

        if not preserve_requests:
            os.remove(requests_filepath)

        if not preserve_results:
            os.remove(save_filepath)
        else:  # Fix requests order
            with open(save_filepath, "r") as f:
                results = [json.loads(line) for line in f]

            sorted_results = sorted(results, key=lambda x: x[2]['original_index'])

            with open(save_filepath, "w") as f:
                for result in sorted_results:
                    json_string = json.dumps(result)
                    f.write(json_string + "\n")

        return validated_data_list


# DashScope基础处理器，支持Qwen大模型对话
class BaseDashscopeProcessor:
    def __init__(self):
        # 从环境变量读取API-KEY
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.default_model = 'qwen-turbo-latest'

    def send_message(
            self,
            model="qwen-turbo-latest",
            temperature=0.1,
            seed=None,  # 兼容参数，暂不使用
            system_content='You are a helpful assistant.',
            human_content='Hello!',
            is_structured=False,
            response_format=None,
            **kwargs
    ):
        """
        发送消息到DashScope Qwen大模型，支持 system_content + human_content 拼接为 messages。
        暂不支持结构化输出。
        """
        if model is None:
            model = self.default_model
        # 拼接 messages
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if human_content:
            messages.append({"role": "user", "content": human_content})
        # print('system_content=', system_content)
        # print('='*30)
        # print('human_content=', human_content)
        # print('='*30)
        # print('messages=', messages)
        # print('='*30)
        # 调用 dashscope Generation.call
        response = dashscope.Generation.call(
            model=model,
            messages=messages,
            temperature=temperature,
            result_format='message'
        )
        print('dashscope.api_key=', dashscope.api_key)
        print('model=', model)
        print('response=', response)
        # 兼容 openai/gemini 返回格式，始终返回 dict
        if hasattr(response, 'output') and hasattr(response.output, 'choices'):
            content = response.output.choices[0].message.content
        else:
            content = str(response)
        # 增加 response_data 属性，保证接口一致性
        self.response_data = {"model": model,
                              "input_tokens": response.usage.input_tokens if hasattr(response, 'usage') and hasattr(
                                  response.usage, 'input_tokens') else None,
                              "output_tokens": response.usage.output_tokens if hasattr(response, 'usage') and hasattr(
                                  response.usage, 'output_tokens') else None}
        print('content=', content)

        # 尝试解析 content 为 JSON，如果是结构化响应
        try:
            # 先尝试移除可能的markdown代码块标记
            content_str = content.strip()
            if content_str.startswith('```') and '```' in content_str[3:]:
                # 找到第一个 ``` 和 最后一个 ``` 之间的内容
                first_backtick = content_str.find('```') + 3
                next_newline = content_str.find('\n', first_backtick)
                if next_newline > 0:
                    first_backtick = next_newline + 1
                last_backtick = content_str.rfind('```')
                if last_backtick > first_backtick:
                    json_str = content_str[first_backtick:last_backtick].strip()
                else:
                    json_str = content_str
            else:
                json_str = content_str

            # 尝试解析 JSON
            parsed_content = json.loads(json_str)
            return parsed_content
        except (json.JSONDecodeError, TypeError):
            # 如果不是有效的JSON，返回基本格式
            print(f"Content is not valid JSON, returning basic format: {content}")
            return {"final_answer": content, "step_by_step_analysis": "", "reasoning_summary": "", "relevant_pages": []}

import json
import pickle
import numpy as np
import faiss
import jieba
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer



class VectorDBIngestor:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 device: Optional[str] = None, batch_size: int = 4):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.vector_size = self.model.get_sentence_embedding_dimension()
        print(f"[VectorDBIngestor] Loaded {model_name}, dim={self.vector_size}, device={self.model.device}")

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        report_paths = sorted(all_reports_dir.glob("*_chunks.json"))
        if not report_paths:
            print("[WARNING] 未找到 *_chunks.json 文件")
            return

        success_count = 0
        skipped_count = 0
        error_count = 0

        for rp in tqdm(report_paths, desc="构建 FAISS 索引"):
            try:
                # 1. 读取元数据计算目标文件名
                with open(rp, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                sha1 = str(data['metainfo'].get('sha1', '')).strip()
                safe_name = sha1 if sha1 and sha1.replace('_','').replace('-','').isalnum() else rp.stem.replace('_chunks', '')
                safe_name = "".join(c for c in safe_name if c.isalnum() or c in ('_', '-'))
                final_path = output_dir / f"{safe_name}.faiss"

                # 🔑 核心：已存在则跳过
                if final_path.exists():
                    print(f"[INFO] ⏭️ 索引已存在，跳过: {final_path.name}")
                    skipped_count += 1
                    continue

                # 2. 检查文本内容
                texts = [c['text'] for c in data['content']['chunks'] if c.get('text')]
                if not texts:
                    continue

                # 3. 分批生成向量
                embeddings = []
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i+self.batch_size]
                    embeddings.extend(self.model.encode(batch, normalize_embeddings=True))

                # 4. 构建 FAISS 索引
                index = faiss.IndexFlatIP(self.vector_size)
                index.add(np.array(embeddings, dtype=np.float32))

                # 5. 绕过 Windows 中文路径 Bug：写入临时 ASCII 路径后移动
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_file = Path(tmpdir) / "tmp_index.faiss"
                    faiss.write_index(index, str(tmp_file))
                    shutil.move(str(tmp_file), str(final_path))

                print(f"[INFO] ✅ 成功保存: {final_path.name}")
                success_count += 1

            except Exception as e:
                print(f"[ERROR] ❌ 处理 {rp.name} 失败: {e}")
                error_count += 1

        # 📊 真实状态汇总
        total = len(report_paths)
        print(f"\n[INFO] 📊 向量库构建完成 | 总计: {total} | ✅ 成功: {success_count} | ⏭️ 跳过: {skipped_count} | ❌ 失败: {error_count}")
        print(f"[INFO] 保存目录: {output_dir}")

# ==================== BM25Ingestor (支持中文分词) ====================
class BM25Ingestor:
    def __init__(self, use_jieba: bool = True):
        self.use_jieba = use_jieba

    def _tokenize(self, text: str) -> List[str]:
        """分词：中文使用 jieba，保留中英文数字混合"""
        if self.use_jieba:
            words = jieba.lcut(text)
            # 过滤纯空格/空字符串
            return [w for w in words if w.strip()]
        else:
            import re
            return re.findall(r'\w+', text.lower())

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        tokenized_chunks = [self._tokenize(chunk) for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="构建 BM25 索引"):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            if not chunks:
                continue
            bm25_index = self.create_bm25_index(chunks)
            sha1 = report_data["metainfo"].get("sha1", report_path.stem)
            output_file = output_dir / f"{sha1}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
        print(f"BM25 索引保存至 {output_dir}")