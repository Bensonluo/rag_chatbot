"""
Rule-based intent detector using keyword matching for business intents.

Fast, lightweight intent detection for customer service scenarios.
"""
import re
from typing import Optional, Dict, List

from app.services.intent.base import IntentDetector, IntentResult
from app.models.enums.intent import Intent


class RuleBasedIntentDetector(IntentDetector):
    """
    Rule-based intent detector using keyword matching for business intents.

    Maps customer service keywords to business intent categories:
    task-oriented (refund, return, order, shipping, complaint),
    knowledge (faq, policy), dialogue (chitchat, greeting),
    and meta intents (confirm, deny, cancel).
    """

    def __init__(self) -> None:
        self.rules = self._build_rules()

    def _build_rules(self) -> Dict[Intent, List[dict]]:
        return {
            # Task-oriented intents
            Intent.REFUND: [
                {"keywords": ["退款", "退钱", "我要退款", "refund", "退费", "申请退款"], "weight": 1.5},
                {"patterns": [r"退(款|钱|费)"], "weight": 1.2},
            ],
            Intent.RETURN: [
                {"keywords": ["退货", "退换", "我要退货", "return", "换货", "退换货"], "weight": 1.5},
                {"patterns": [r"退[换货]"], "weight": 1.2},
            ],
            Intent.QUERY_ORDER: [
                {"keywords": ["订单", "查订单", "我的订单", "order", "订单状态", "订单查询", "查一下订单"], "weight": 1.5},
                {"patterns": [r"订单.{0,4}(状态|查询|情况|在哪)"], "weight": 1.3},
            ],
            Intent.TRACK_SHIPPING: [
                {"keywords": ["物流", "快递", "到哪了", "配送", "shipping", "发货", "运单", "快递单号"], "weight": 1.5},
                {"patterns": [r"(物流|快递|包裹).{0,4}(到|在|哪|状态|查询)"], "weight": 1.3},
            ],
            Intent.COMPLAINT: [
                {"keywords": ["投诉", "差评", "不满", "complaint", "举报", "投诉客服", "不满意"], "weight": 1.5},
            ],
            # Knowledge intents
            Intent.FAQ: [
                {"keywords": ["怎么", "如何", "为什么", "能不能", "可以", "是否"], "weight": 0.5},
                {"patterns": [
                    r"(怎么|如何).{1,10}(退|退款|退货|换货)",
                    r"(运费|邮费|配送费).{0,4}(多少|怎么算|免)",
                ], "weight": 1.2},
            ],
            Intent.POLICY: [
                {"keywords": ["政策", "规定", "规则", "条款", "保障", "协议"], "weight": 1.0},
                {"patterns": [
                    r"(退换|退货|退款|售后).{0,6}(政策|规定|规则|条件|是什么|怎么算)",
                    r"(政策|规定|规则|条件).{0,4}(退|换|退款)",
                    r".{0,4}(政策|规定|规则).{0,6}(是什么|有哪些|怎么样)",
                    r"(保修|质保|售后).{0,4}(期|政策|规定)",
                ], "weight": 2.0},
            ],
            # Dialogue intents
            Intent.CHITCHAT: [
                {"keywords": [
                    "哈哈", "呵呵", "搞笑", "笑话", "无聊", "天气",
                    "周末", "吃什么", "推荐电影", "玩游戏",
                ], "weight": 0.8},
            ],
            Intent.GREETING: [
                {"keywords": [
                    "你好", "在吗", "hello", "hi", "hey",
                    "您好", "早上好", "下午好", "晚上好",
                    "good morning", "good afternoon",
                ], "weight": 1.5},
                {"patterns": [r"^(你好|您好|hi|hello|hey)[!！。]*$"], "weight": 1.5},
            ],
            # Meta intents
            Intent.CONFIRM: [
                {"keywords": ["是的", "对", "确认", "好的", "没错", "正确", "可以", "继续", "yes", "确定", "要的"], "weight": 1.5},
                {"patterns": [r"^(是的|对|好的|确认|没错|可以|继续|要的)[！!。]*$"], "weight": 1.3},
            ],
            Intent.DENY: [
                {"keywords": ["不是", "不对", "不要", "不行", "不可以", "no", "没有", "不是的"], "weight": 1.5},
                {"patterns": [r"^(不是|不对|不要|不行|没有)[！!。]*$"], "weight": 1.3},
            ],
            Intent.CANCEL: [
                {"keywords": ["取消", "算了", "不要了", "cancel", "取消退款", "取消退货", "不办了"], "weight": 1.5},
                {"patterns": [r"^(取消|算了|不要了|不办了)[！!。]*$"], "weight": 1.3},
            ],
        }

    def detect(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> Intent:
        result = self.detect_with_confidence(query, context)
        return result.intent

    def detect_with_confidence(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> IntentResult:
        if not query or not query.strip():
            return IntentResult(intent=Intent.UNKNOWN, confidence=0.0)

        normalized_query = self._normalize_query(query)
        scores: Dict[Intent, float] = {intent: 0.0 for intent in Intent}
        matched_rules: list = []

        for intent, rules in self.rules.items():
            for rule in rules:
                if "keywords" in rule:
                    matched = self._contains_any(normalized_query, rule["keywords"])
                    if matched:
                        scores[intent] += rule["weight"]
                        matched_rules.append({
                            "intent": intent.value,
                            "rule_type": "keyword",
                            "weight": rule["weight"],
                        })

                if "patterns" in rule:
                    for pattern in rule["patterns"]:
                        if re.search(pattern, normalized_query, re.IGNORECASE):
                            scores[intent] += rule["weight"]
                            matched_rules.append({
                                "intent": intent.value,
                                "rule_type": "pattern",
                                "pattern": pattern,
                                "weight": rule["weight"],
                            })

        max_intent = Intent.UNKNOWN
        max_score = 0.0

        for intent, score in scores.items():
            if score > max_score:
                max_score = score
                max_intent = intent

        confidence = min(max_score / 3.0, 1.0) if max_score > 0 else 0.0

        return IntentResult(
            intent=max_intent,
            confidence=confidence,
            metadata={
                "matched_rules": matched_rules,
                "raw_score": max_score,
            } if matched_rules else None,
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        return query.lower().strip()

    @staticmethod
    def _contains_any(text: str, keywords: list) -> bool:
        return any(kw.lower() in text for kw in keywords)
