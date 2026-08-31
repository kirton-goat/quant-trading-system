from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from openai import OpenAI

from config import Settings
from market import MarketSnapshot, NewsItem


class AiEngine:
    def __init__(self, settings: Settings) -> None:
        self.enabled = bool(settings.deepseek_api_key)
        self.client = (
            OpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                timeout=settings.request_timeout,
            )
            if self.enabled
            else None
        )

    def extract_stock_code(self, news: NewsItem) -> str | None:
        if not self.enabled or self.client is None:
            print("未设置 DEEPSEEK_API_KEY，跳过AI识别股票。")
            return None

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是精通A股市场复杂关联的顶级投资经理。阅读新闻时，"
                        "不要只做字面提取，必须进行实体对齐与商业映射推理。\n"
                        "输出规则：\n"
                        "1. 如果能映射到明确A股上市公司，只返回一个最核心的6位股票代码。\n"
                        "2. 如果只是概念/产业链机会，不能确定唯一股票代码，则返回一个简短主题标签，"
                        "例如：DeepSeek概念股、AI算力概念股、华为汽车产业链、小米汽车产业链。\n"
                        "3. 如果新闻和A股交易机会弱相关，返回 None。\n"
                        "关键映射：\n"
                        "- 问界、AITO、M7、M9、余承东、华为智选车：优先映射赛力斯 601127。\n"
                        "- 智界：优先映射奇瑞产业链/华为汽车产业链；无法确定唯一代码时返回 华为汽车产业链。\n"
                        "- 享界：优先映射北汽蓝谷 600733。\n"
                        "- 小米汽车、SU7、雷军：无法确定唯一代码时返回 小米汽车产业链。\n"
                        "- DeepSeek、深度求索、DSpark：若是技术突破、合作、生态扩张，"
                        "返回 DeepSeek概念股，不要返回 None。\n"
                        "只返回代码、主题标签或 None，不要解释。"
                    ),
                },
                {"role": "user", "content": f"标题：{news.title}\n正文：{news.content}"},
            ],
            stream=False,
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r"\d{6}", text)
        if match:
            return match.group(0)
        if text.lower() == "none":
            return None
        return text[:40] if text else None

    def decide(self, news: NewsItem, code: str | None, snapshot: MarketSnapshot | None) -> dict[str, Any]:
        if not self.enabled or self.client is None:
            return {
                "score": 0,
                "action": "观望",
                "logic": "未设置 DEEPSEEK_API_KEY，无法调用AI。当前仅完成新闻/行情框架检查。",
            }

        market_payload = asdict(snapshot) if snapshot else {}
        prompt = f"""
你是稳健型A股量化交易助手。请根据新闻、实体映射结果和行情数据输出交易建议。

要求：
1. 只输出 JSON，不要 Markdown。
2. action 只能是：买入、卖出、观望。
3. score 为 0-10 分。
4. logic 不超过 80 个中文字符。
5. 如果识别结果是具体6位A股代码，要结合行情趋势判断。
6. 如果识别结果是概念标签，例如 DeepSeek概念股、华为汽车产业链、小米汽车产业链，
   要说明概念关联原因；没有具体股票行情时，操作一般保持观望，除非新闻极强且逻辑清晰。
7. 新闻不明确、数据缺失、趋势走坏时优先观望。

新闻标题：{news.title}
新闻正文：{news.content}
识别结果：{code or "未识别"}
行情数据：{json.dumps(market_payload, ensure_ascii=False)}
"""
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        raw = response.choices[0].message.content.strip()
        return self._parse_json_decision(raw)

    @staticmethod
    def _parse_json_decision(raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"score": 0, "action": "观望", "logic": raw[:160]}

        return {
            "score": data.get("score", 0),
            "action": data.get("action", "观望"),
            "logic": data.get("logic", ""),
        }
