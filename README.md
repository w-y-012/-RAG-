RAG 智能财报问答系统
基于 检索增强生成（RAG） 技术的财报问答系统，专为中芯国际等上市公司年报设计。支持向量检索、BM25 关键词检索、混合检索（RRF 融合）以及大模型重排序，可回答数值、布尔、实体列表、开放性文本等多种类型的问题。


项目简介
本项目实现了一个完整的 RAG 问答流水线，输入为 PDF 格式的年报，输出为针对自然语言问题的结构化答案。系统包含以下关键环节：

PDF 解析：使用 MinerU 将 PDF 转换为结构化的 content_list.json。

智能分块：基于标题层级、表格、列表等语义边界进行分块，并记录跨页页码。

索引构建：生成 FAISS 向量索引（384 维轻量模型）和 BM25 关键词索引。

混合检索：同时执行向量检索与 BM25 检索，通过 RRF 融合排序。

LLM 重排序（可选）：使用大模型对初检结果进行二次精排，提升相关性。

答案生成：将检索到的上下文与问题一同提交给 LLM，生成结构化答案（支持 number/boolean/names/string/comparative 等多种格式）。


pip install -r requirements.txt
主要依赖包：

sentence-transformers：本地嵌入模型（paraphrase-multilingual-MiniLM-L12-v2）

faiss-cpu：向量索引

rank-bm25：BM25 索引

jieba：中文分词

langchain-text-splitters：文本分割

tiktoken：Token 计数

pandas、tqdm 等


数据准备
1. 放置 PDF 文件
将目标财报 PDF 放入 data/stock_data/pdf_reports/ 目录（可自定义）。

2. 准备 subset.csv
文件格式（UTF-8 或 GBK）：

csv
sha1,file_name,company_name
stock_10001,【财报】中芯国际：中芯国际2024年年度报告,中芯国际
sha1：唯一标识，用于索引文件命名。

file_name：PDF 文件名（不含扩展名）。

company_name：公司全称，用于问题中公司名匹配。

3. 准备问题文件 questions.json
json
[
    {
        "text": "中芯国际2024年营业收入是多少？",
        "kind": "number"
    },
    {
        "text": "公司是否在2024年进行了分红？",
        "kind": "boolean"
    },
    {
        "text": "列出2024年公司的前五大客户名称。",
        "kind": "names"
    }
]
支持的 kind 类型：number、boolean、names、string、comparative。

运行流程
完整流水线（一键执行）
bash
python src/pipeline.py
默认配置（max_config）会依次执行：PDF 解析 → 分块 → 构建向量库 → 构建 BM25 → 问答。

分步执行（调试用）
可在 if __name__ == "__main__" 中单独运行各步骤：

python
pipeline = Pipeline(root_path, run_config=max_config)

# 1. 将 PDF 转换为 content_list.json（需要 MinerU API Key）
# pipeline.export_reports_to_markdown('【财报】中芯国际：中芯国际2024年年度报告.pdf')

# 2. 分块
pipeline.chunk_reports()

# 3. 构建向量库（FAISS）
pipeline.create_vector_dbs()

# 4. 构建 BM25 索引
pipeline.create_bm25_db()

# 5. 回答所有问题
pipeline.process_questions()
单问题交互
python
answer = pipeline.answer_single_question("中芯国际2024年净利润是多少？", kind="number")
print(answer["final_answer"])
模拟模式（无真实 LLM API）
在 .env 或代码中设置：

python
os.environ["USE_MOCK_LLM"] = "true"
此时所有 LLM 调用返回固定占位答案，用于测试检索和分块逻辑。

核心模块详解
1. 智能分块器（TextSplitter）
输入：MinerU 输出的 *_content_list.json

输出：*_chunks.json（包含每个 chunk 的文本、页码、章节标题、表格结构等）

特性：

自动关联多级标题（section_title）

连续文本合并（直到遇到表格/图片等特殊块）

跨页页码记录（pages 列表）

表格转换为 Markdown + 结构化元数据

2. 检索器
向量检索（VectorRetriever）：使用 FAISS + 本地嵌入模型（384 维）

BM25 检索（BM25Retriever）：基于 jieba 分词，适合关键词匹配

混合检索（HybridRetriever）：同时执行上述两种检索，通过 RRF 融合排序，可选 LLM 重排

3. 答案生成（APIProcessor）
根据 kind 自动选择提示模板和输出格式（Pydantic schema）


使用手册
自定义配置
修改 RunConfig 实例（如 max_config）：

eval_data_answers_{config_suffix}.json：RAGAS 评估数据（若启用）
