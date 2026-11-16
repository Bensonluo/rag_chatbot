"""Conservative request boundaries and safety guards for opt-in multi detection.

These heuristics deliberately abstain on ambiguous syntax. They are not a
semantic parser and do not authorize commerce operations.
"""

import re

from app.models.enums.intent import Intent

# Mask quotes without changing offsets. Apostrophes in contractions are not quotes.
_QUOTES = re.compile(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」|(?<!\w)\'[^\'\n]*\'')
_CORRECTION = re.compile(
    r"\b(?:actually|instead|i mean|rather than|no[, ]+i mean)\b|"
    r"[,，]\s*no\s*[,，]|不对[，,\s]*|改为|改成|不是.{0,20}而是",
    re.IGNORECASE,
)
_REQUEST_START = (
    r"(?:(?:please\s+)?(?:i\s+(?:want|would|need)|refund\b|return\b|cancel\b|"
    r"track\b|check\b|query\b|tell\b|what\b|how\b|where\b|can\s+you\b|connect\b)|"
    r"(?:我要|我想|帮我|请|退货|退款|查订单|查物流|转人工|运费|怎么|如何|什么))"
)
_BOUNDARY = re.compile(
    r"[;；。\n!?！？]+|\.(?=\s+[A-Z])|"
    r"\b(?:and\s+then|and\s+also|and|also|plus|but)\s+(?=" + _REQUEST_START + r")|"
    r"(?:另外|顺便|同时|还有|以及|然后|并且)|"
    r"[,，]\s*(?=" + _REQUEST_START + r")",
    re.IGNORECASE,
)
_BLOCKED = re.compile(
    r"\b(?:do\s+not|don't|dont|never|not\s+now|maybe|later|tomorrow|if|"
    r"not\s+(?:asking|requesting|seeking|wanting|looking|going|planning|interested)\b|"
    r"unless|provided\s+(?:that|it)|assuming|on\s+condition\s+that|"
    r"not\s+(?:to\s+)?(?:refund|return|cancel)|no\s+(?:refund|return))\b|"
    r"不要|不想|不需要|别|暂时|稍后|以后|明天|如果|先不|再说",
    re.IGNORECASE,
)


def visible_text(text: str) -> str:
    """Hide quoted content while preserving the original character offsets."""
    return _QUOTES.sub(lambda match: " " * len(match.group()), text)


def is_blocked_request(text: str) -> bool:
    """Conservatively exclude negated, conditional and deferred clauses."""
    return bool(_BLOCKED.search(visible_text(text)))


def request_ranges(query: str) -> list[tuple[int, int]]:
    """Return source offsets, preserving occurrences and correction scope."""
    visible = visible_text(query)
    corrections = list(_CORRECTION.finditer(visible))
    start = corrections[-1].end() if corrections else 0
    ranges = []

    def append(left: int, right: int) -> None:
        while left < right and query[left] in " ,，;；\n\t":
            left += 1
        while right > left and query[right - 1] in " ,，;；\n\t":
            right -= 1
        if left < right:
            ranges.append((left, right))

    for boundary in _BOUNDARY.finditer(visible, start):
        connector = boundary.group().strip().lower()
        # Descriptive lead-ins belong to the following request. A
        # conjunction alone does not turn its left side into a task.
        contextual_separator = connector.startswith((",", "，")) or connector in {
            "and",
            "and also",
            "and then",
            "but",
            "plus",
            "also",
        }
        # A condition owns its consequent, including one introduced by a
        # comma or conjunction; splitting it would manufacture an action.
        if contextual_separator and _CONDITIONAL_ACT.search(visible[start : boundary.start()]):
            continue
        if (
            contextual_separator
            and visible[start : boundary.start()].strip()
            and not re.search(
                _REQUEST_START + r"|[?？]|到哪|多久|几天|什么时候|何时",
                visible[start : boundary.start()],
                re.IGNORECASE,
            )
        ):
            continue
        if connector in {
            "and",
            "and also",
            "and then",
            "plus",
            "also",
            "以及",
            "同时",
            "并且",
        } and is_blocked_request(query[start : boundary.start()]):
            continue
        # Sentence terminators belong to the request; conjunctions do not.
        end = boundary.end() if connector and connector[0] in ".!?。！？" else boundary.start()
        append(start, end)
        start = boundary.end()
    append(start, len(query))
    return ranges


def request_spans(query: str) -> list[str]:
    """Return verbatim independent clauses, retaining repeated intent requests.

    A correction supersedes preceding text. Commas only split before explicit
    request cues; quoted separators and entity-description commas stay intact.
    """
    return [query[left:right] for left, right in request_ranges(query)]


def is_informational_action(text: str) -> bool:
    """Exclude policy/how-to language from executable refund/return/cancel spans."""
    return bool(
        re.search(
            r"\b(?:how|polic(?:y|ies)|conditions?|rules?)\b|"
            r"\bwhat\s+(?:does|do)\b.{0,80}\bmean\b|"
            r"\bwhat\s+(?:is|are)\s+(?:(?:a|an|the)\s+)?"
            r"(?:refund|return|exchange|cancellation)\b|"
            r"\b(?:explain|define|definition|meaning)\b.{0,80}"
            r"\b(?:refund|return|exchange|cancel)\w*\b|"
            r"政策|规定|规则|条件|怎么|如何|多久|几天|什么时候|何时|"
            r"什么意思|什么含义|含义|定义|解释.{0,20}(?:退|换|取消)",
            visible_text(text),
            re.IGNORECASE,
        )
    )


def has_unquoted_action(text: str) -> bool:
    """Quoted action mentions alone are not evidence of a current request."""
    visible = visible_text(text)
    if visible == text:
        return True
    return bool(
        re.search(
            r"\b(?:refund|reimburse|return|exchange|cancel|confirm|track|check|"
            r"complain|complaint|human|agent)\b|money\s+back|"
            r"退款|退钱|退费|退货|换货|取消|确认|查|投诉|人工",
            visible,
            re.IGNORECASE,
        )
    )


def plausible_mixed_request(query: str) -> bool:
    """A strong first clause must not hide a second, even unrecognized, clause."""
    return len(request_spans(query)) > 1


# Speech-act markers describe the request, not incidental order chronology.
# Conditional scope precedes completed-action evidence; "already shipped"
# inside a condition must not turn the requested refund into history.
_ALTERNATIVE_ACT = re.compile(
    r"\bshould\s+i\b.{0,40}\bor\b"
    r"|\b(?:return|refund|cancel|exchange|keep)\s+or\s+(?:return|refund|cancel|exchange|keep)\b"
    r"|\bwhich\s+(?:one\s+)?(?:is\s+)?(?:better|best|cheaper|wiser)\b"
    r"|\bbetter\s+to\s+(?:return|refund|cancel|exchange|keep)\b"
    r"|\b(?:what\s+do\s+you\s+think|do\s+you\s+think\s+i\s+should|how\s+about)\b"
    r"|还是.{0,12}(?:好|划算|合适|值得|好一点|好些)"
    r"|(?:退货|退款|换货|取消)还是|该不该(?:退|换|取消)|要不要(?:退|换|取消)"
    r"|你觉得|你建议|参谋一下",
    re.IGNORECASE,
)
_HISTORICAL_ACT = re.compile(
    r"\b(?:returned|refunded|cancelled|canceled|exchanged|complained|checked|tracked|queried)\b"
    r"|\b(?:requested|asked\s+for|applied\s+for|received|got)\b.{0,30}"
    r"\b(?:refund|return|cancellation)\b"
    r"|\b(?:refund|return|cancellation)\b.{0,40}\b(?:was|has\s+been|had\s+been)\s+"
    r"(?:already\s+)?(?:processed|completed|approved|issued)\b"
    r"|(?:退[款货钱费]?|换货?|取消|投诉|申请|查(?:询)?(?:过)?(?:订单|物流)?)过"
    r"|(?:已经|早就|之前|已)(?:退|换|取消|投诉|申请|查)"
    r"|(?:退[款货钱费]?|换货?|取消|投诉|申请|查(?:询)?(?:订单|物流)?)(?:了)",
    re.IGNORECASE,
)
_CURRENT_REQUEST_ACT = re.compile(
    r"^\s*(?:(?:please|just)\s+)*(?:refund|return|exchange|cancel|track|check|query)\b"
    r"|^\s*(?:(?:can|could|would|will)\s+you\s+|"
    r"i\s+(?:want|need|would\s+like)\s+(?:to\s+)?(?:a\s+)?)"
    r"(?:please\s+)?(?:refund|return|exchange|cancel|track|check|query)\b"
    r"|(?:^|[,，])\s*(?:请|帮我|我要|我想|麻烦).{0,10}(?:退|换|取消|查)",
    re.IGNORECASE,
)
_REPORTED_ACT = re.compile(
    r"^\s*(?:(?:the|an?|my|your)\s+)?(?:email|message|letter|agent|seller|support|"
    r"representative|friend|he|she|they)\s+(?:mentions?|mentioned|says?|said|"
    r"suggests?|suggested|wrote|recommends?|recommended)\b"
    r"|^\s*(?:邮件|消息|短信|客服|卖家|他|她)(?:里|中)?(?:说|写|提到|建议|推荐)",
    re.IGNORECASE,
)
_CONDITIONAL_ACT = re.compile(
    r"\bif\b|\bunless\b|\bprovided\s+that\b|\bas\s+long\s+as\b"
    r"|如果|假如|要是|若是",
    re.IGNORECASE,
)
# A clause that both carries a condition and defers the action to later is
# a deferral ("if possible, refund tomorrow"), not a request to verify a
# condition now. It stays suppressed like other non-current mentions.
_DEFERRED_ACT = re.compile(
    r"\btomorrow\b|\blater\b|\bnext\s+(?:week|month)\b|先不|再说|暂时|稍后|以后|明天",
    re.IGNORECASE,
)


def classify_speech_act(text: str) -> str:
    """Classify how a clause is meant, conservative on executable reads.

    Historical/alternative/conditional labels let the admission layer keep
    history mentions, advice questions and unverified conditions away from
    task creation without discarding the clause's information.
    """
    visible = visible_text(text)
    if _ALTERNATIVE_ACT.search(visible):
        return "alternative"
    if _CONDITIONAL_ACT.search(visible) and not _DEFERRED_ACT.search(visible):
        return "conditional"
    if is_informational_action(visible) or (
        _REPORTED_ACT.search(visible) and has_unquoted_action(text)
    ):
        return "informational"
    if _CURRENT_REQUEST_ACT.search(visible):
        return "action"
    if _HISTORICAL_ACT.search(visible):
        return "historical"
    return "action"


def shared_target_count(intent: Intent, text: str, speech_act: str) -> int:
    """Expand only current per-order tasks, never topics or meta cancellation.

    The model emits one evidence frame; both detectors assign entity indices
    here exactly once. Complaints and handoffs can mention multiple orders
    without requesting multiple independent operations.
    """
    from app.services.slot_filling.slot_types import find_order_ids

    if speech_act == "action" and intent in {
        Intent.REFUND,
        Intent.RETURN,
        Intent.QUERY_ORDER,
        Intent.TRACK_SHIPPING,
    }:
        return max(1, len(find_order_ids(text)))
    return 1
