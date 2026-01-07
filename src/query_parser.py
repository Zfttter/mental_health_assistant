import json
import re
from dataclasses import dataclass, asdict
from typing import Callable, Optional


INTENTS = ("advice", "definition", "symptom", "emergency", "treatment", "self_help", "other")
TOPICS = (
    "sleep",
    "anxiety",
    "depression",
    "stress",
    "panic",
    "therapy",
    "medication",
    "relationships",
    "trauma",
    "work",
    "other",
)


@dataclass
class ParsedQuery:
    intent: str
    topic: str
    keywords: list[str]
    language: str = "auto"

    def to_json(self) -> dict:
        return asdict(self)


_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[\u4e00-\u9fff]{1,}")


def _simple_keywords(text: str, max_keywords: int = 8) -> list[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    # very small stop list (keep it tiny to avoid false negatives)
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "am",
        "i",
        "me",
        "my",
        "you",
        "your",
        "what",
        "how",
        "why",
        "should",
        "can",
        "could",
        "would",
        "please",
    }
    keywords = []
    for t in tokens:
        if t in stop:
            continue
        if t.isdigit():
            continue
        if t not in keywords:
            keywords.append(t)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _rule_intent_topic(query: str) -> tuple[str, str]:
    q = query.lower()

    # emergency heuristics
    emergency_markers = [
        "suicide",
        "kill myself",
        "end my life",
        "self-harm",
        "hurt myself",
        "overdose",
        "emergency",
        "call 911",
        "call 112",
        "cannot breathe",
        "chest pain",
        "hearing voices",
        "psychosis",
    ]
    if any(m in q for m in emergency_markers):
        return "emergency", "other"

    if any(m in q for m in ["what is", "define", "definition of", "meaning of"]):
        intent = "definition"
    elif any(m in q for m in ["symptom", "signs", "do i have", "diagnose"]):
        intent = "symptom"
    elif any(m in q for m in ["treatment", "medication", "therapy", "ssri", "cbt"]):
        intent = "treatment"
    elif any(m in q for m in ["tips", "how to", "help", "cope", "manage", "advice"]):
        intent = "advice"
    else:
        intent = "other"

    topic_map = {
        "sleep": ["sleep", "insomnia", "nightmare"],
        "anxiety": ["anxiety", "anxious", "worry", "worried"],
        "depression": ["depression", "depressed", "sad", "hopeless"],
        "stress": ["stress", "stressed", "burnout"],
        "panic": ["panic", "panic attack"],
        "therapy": ["therapy", "therapist", "counseling", "counsellor"],
        "medication": ["medication", "antidepressant", "ssri", "prozac", "sertraline"],
        "relationships": ["relationship", "partner", "family", "friend"],
        "trauma": ["trauma", "ptsd", "flashback"],
        "work": ["work", "job", "workplace"],
    }
    for topic, markers in topic_map.items():
        if any(m in q for m in markers):
            return intent, topic

    return intent, "other"


def parse_query(
    query: str,
    llm_json: Optional[Callable[[str], str]] = None,
    llm_model_hint: str = "gemma2-9b-it",
) -> ParsedQuery:
    """
    Parse query into intent/topic/keywords WITHOUT training.

    - Prefer LLM JSON parsing if llm_json is provided.
    - Fallback to deterministic rule-based parsing if LLM fails.
    """
    # 1) LLM-based parsing (optional)
    if llm_json is not None:
        prompt = f"""
You are a query understanding module for a mental-health retrieval system.
Return ONLY valid JSON with the following schema:
{{
  "intent": one of {list(INTENTS)},
  "topic": one of {list(TOPICS)},
  "keywords": array of 3-10 short keywords/phrases extracted from the query,
  "language": "en" | "zh" | "auto"
}}

Rules:
- Choose "emergency" if the user indicates self-harm, suicide, severe danger, overdose, or immediate crisis.
- Keep keywords faithful to the query (no new medical claims).
- Do not include any extra keys.

Query: {query}
""".strip()
        try:
            raw = llm_json(prompt)
            parsed = json.loads(raw)
            intent = str(parsed.get("intent", "")).lower()
            topic = str(parsed.get("topic", "")).lower()
            keywords = parsed.get("keywords", [])
            language = str(parsed.get("language", "auto")).lower()

            if intent not in INTENTS:
                raise ValueError(f"invalid intent: {intent}")
            if topic not in TOPICS:
                topic = "other"
            if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
                keywords = _simple_keywords(query)
            keywords = [k.strip().lower() for k in keywords if k.strip()][:10]
            if len(keywords) < 3:
                keywords = _simple_keywords(query)
            if language not in ("en", "zh", "auto"):
                language = "auto"
            return ParsedQuery(intent=intent, topic=topic, keywords=keywords, language=language)
        except Exception:
            # fallback below
            pass

    # 2) Rule-based parsing (deterministic)
    intent, topic = _rule_intent_topic(query)
    return ParsedQuery(intent=intent, topic=topic, keywords=_simple_keywords(query), language="auto")

