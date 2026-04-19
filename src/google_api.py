import asyncio
import logging
import os
import re
from typing import Dict, List, Tuple
from core_security import TraceContext

logger = logging.getLogger("SecureRuntime")


class GoogleSearchTool:
    """
    ✅ 搜索引擎工具 - 适配 serpapi 1.0.2
    L1: SerpApi (付费，稳定)
    L2: DuckDuckGo (免费)
    L3: Mock (兜底)
    """

    def __init__(self, api_key: str = "", mock_mode: bool = False):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY", "")
        self.mock_mode = mock_mode
        self._cache: Dict[str, Dict] = {}

        if self.mock_mode:
            logger.info("🌐 搜索工具初始化 | 模式：MOCK")
        elif self.api_key:
            logger.info("🌐 搜索工具初始化 | 模式：SerpApi 1.x")
        else:
            logger.info("🌐 搜索工具初始化 | 模式：DuckDuckGo (免费)")

    async def search(self, query: str, num_results: int = 5) -> Dict:
        """执行搜索 - 自动降级"""
        cache_key = f"{query}_{num_results}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.mock_mode:
            return self._mock_search(query, num_results)

        # L1: SerpApi
        if self.api_key:
            result = await self._serpapi_search(query, num_results)
            if result and result.get("results"):
                self._cache[cache_key] = result
                return result

        # L2: DuckDuckGo
        result = await self._duckduckgo_search(query, num_results)
        if result and result.get("results"):
            self._cache[cache_key] = result
            return result

        # L3: Mock
        return self._mock_search(query, num_results)

    async def _serpapi_search(self, query: str, num_results: int) -> Dict:
        """
        ✅ L1: SerpApi 搜索 (适配 1.0.2 版本)
        使用 serpapi.search() 函数直接调用
        """
        try:
            import serpapi

            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "gl": "cn",
                "hl": "zh-cn",
                "num": num_results
            }

            # ✅ serpapi 1.x 正确用法：直接调用 search 函数
            results = await asyncio.to_thread(lambda: serpapi.search(params))

            formatted = []

            # 优先答案框
            if "answer_box" in results:
                ab = results["answer_box"]
                formatted.append({
                    "title": ab.get("title", "直接答案"),
                    "snippet": ab.get("answer", ab.get("snippet", "")),
                    "link": ab.get("link", "#"),
                    "source": "serpapi_answer"
                })

            # 有机结果
            for res in results.get("organic_results", [])[:num_results]:
                formatted.append({
                    "title": res.get("title", ""),
                    "snippet": res.get("snippet", ""),
                    "link": res.get("link", ""),
                    "source": "serpapi"
                })

            if formatted:
                logger.info(f"🔍 SerpApi | {query[:40]}... | {len(formatted)} 条")
                return {"results": formatted[:num_results], "query": query, "source": "serpapi"}

            return None

        except Exception as e:
            logger.warning(f"⚠️ SerpApi 失败：{e}")
            return None

    async def _duckduckgo_search(self, query: str, num_results: int) -> Dict:
        """✅ L2: DuckDuckGo 搜索（免费）"""
        try:
            from duckduckgo_search import DDGS

            results = await asyncio.to_thread(
                lambda: list(DDGS().text(query, max_results=num_results))
            )

            if results:
                formatted = [
                    {
                        "title": r.get("title", "")[:100],
                        "snippet": r.get("body", ""),
                        "link": r.get("href", ""),
                        "source": "duckduckgo"
                    }
                    for r in results[:num_results]
                ]
                logger.info(f"🔍 DuckDuckGo | {query[:40]}... | {len(formatted)} 条")
                return {"results": formatted, "query": query, "source": "duckduckgo"}

            return None

        except Exception as e:
            logger.warning(f"⚠️ DuckDuckGo 失败：{e}")
            return None

    def _mock_search(self, query: str, num_results: int) -> Dict:
        """✅ L3: Mock 兜底"""
        results = [
            {
                "title": f"搜索结果 {i + 1}: {query[:50]}",
                "snippet": f"这是关于'{query[:30]}...'的模拟结果",
                "link": f"https://example.com/{i}",
                "source": "mock"
            }
            for i in range(min(num_results, 3))
        ]
        logger.warning(f"⚠️ Mock 兜底 | {query[:40]}...")
        return {"results": results, "query": query, "source": "mock"}

    # 🔑 信源权威性白名单（可根据业务扩展）
    AUTHORITY_DOMAINS = {
        "sina.com.cn": 0.9, "eastmoney.com": 0.95, "cls.cn": 0.85, "wallstreetcn.com": 0.9,
        "sec.gov": 1.0, "hkexnews.hk": 1.0, "cninfo.com.cn": 1.0, "pbc.gov.cn": 1.0,
        "gov.cn": 1.0, "xinhuanet.com": 0.95, "people.com.cn": 0.95, "sohu.com": 0.6,
        "zhihu.com": 0.5, "baidu.com": 0.4, "baike.baidu.com": 0.6
    }

    async def verify_fact(self, question: str, rag_answer: str, rag_context: str) -> Tuple[float, str, List[str]]:
        """
        🔍 事实验证：基于网络搜索结果对 RAG 答案进行多维置信度评估
        :return: (score: 0~1, verdict: str, matched_sources: List[str])
        """
        if not rag_answer or len(rag_answer.strip()) < 15 or "error" in rag_answer.lower():
            return 0.0, "答案无效或过短", []

        # 1. 构建精准验证 Query（避免搜无关信息）
        key_facts = self._extract_key_facts(rag_answer)
        search_query = f"{question} {' '.join(key_facts[:3])}"  # 限制长度防截断

        # 2. 执行搜索
        search_res = await self.search(search_query, num_results=5)
        if not search_res or not search_res.get("results"):
            return 0.0, "网络无相关信源", []

        snippets = search_res["results"]

        # 3. 多维度计算置信度
        coverage = self._calc_coverage(rag_answer, snippets, key_facts)
        authority = self._calc_authority(snippets, rag_answer)
        consistency = self._calc_consistency(snippets, rag_answer)

        final_score = coverage * 0.4 + authority * 0.3 + consistency * 0.3
        final_score = min(max(final_score, 0.0), 1.0)  # 钳制在 0~1

        # 4. 判定与提取匹配源
        matched = [s["title"] for s in snippets if self._contains_fact(s["snippet"], rag_answer)]
        if final_score >= 0.75:
            verdict = "✅ 高度可信 (多权威信源交叉验证)"
        elif final_score >= 0.45:
            verdict = "⚠️ 部分存疑 (信源冲突或覆盖不足)"
        else:
            verdict = "❌ 置信度低 (与主流信源不符或未检索到)"

        logger.info(f"🔍 验证 | 分数: {final_score:.2f} | {verdict} | 匹配源: {len(matched)}")
        return final_score, verdict, matched

    # ================= 辅助计算引擎 =================
    def _extract_key_facts(self, text: str) -> List[str]:
        """提取答案中的关键实体/数字/短语（免 jieba 依赖）"""
        # 匹配：中文词(≥2字) | 英文/数字串(≥2位) | 日期/金额格式
        pattern = r'[\u4e00-\u9fa5]{2,}|[a-zA-Z0-9]{2,}|\d+[.%亿万元]?'
        candidates = re.findall(pattern, text)
        # 过滤停用词
        stop_words = {"的", "是", "在", "和", "与", "及", "等", "都", "了", "为", "有", "公司", "股份", "有限"}
        return [c for c in candidates if c not in stop_words]

    def _calc_coverage(self, answer: str, snippets: List[Dict], facts: List[str]) -> float:
        """事实覆盖度：答案中的关键信息有多少被网络信源提及"""
        if not facts: return 0.5
        matched_facts = 0
        for f in facts:
            if any(f.lower() in s["snippet"].lower() for s in snippets):
                matched_facts += 1
        return matched_facts / len(facts)

    def _calc_authority(self, snippets: List[Dict], answer: str) -> float:
        """信源权威性：匹配答案的信源中，高权重域名占比"""
        if not snippets: return 0.0
        weights = []
        for s in snippets:
            if self._contains_fact(s["snippet"], answer):
                domain = s.get("link", "").split('/')[2] if "link" in s else ""
                # 精确匹配 > 模糊匹配
                w = self.AUTHORITY_DOMAINS.get(domain, 0.3)
                weights.append(w)
            else:
                weights.append(0.0)
        return sum(weights) / len(weights) if weights else 0.0

    def _calc_consistency(self, snippets: List[Dict], answer: str) -> float:
        """交叉一致性：多个独立信源是否陈述相同事实"""
        if len(snippets) < 2: return 0.3
        match_count = sum(1 for s in snippets if self._contains_fact(s["snippet"], answer))
        return match_count / len(snippets)

    def _contains_fact(self, snippet: str, answer: str) -> bool:
        """判断 snippet 是否包含 answer 的核心语义（支持模糊包含）"""
        if not snippet: return False
        sn_low, an_low = snippet.lower(), answer.lower()
        # 核心词包含度 > 60% 即认为匹配
        core_words = [w for w in self._extract_key_facts(answer) if len(w) > 2]
        if not core_words: return False
        matches = sum(1 for w in core_words if w in sn_low)
        return matches / len(core_words) >= 0.6

    def clear_cache(self):
        self._cache.clear()

    def get_cache_stats(self) -> Dict:
        return {
            "cache_size": len(self._cache),
            "mock_mode": self.mock_mode,
            "api_key_configured": bool(self.api_key)
        }
