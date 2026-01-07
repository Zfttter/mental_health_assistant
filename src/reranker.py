import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional


_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[\u4e00-\u9fff]{1,}")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _keyword_coverage(keywords: list[str], text: str) -> tuple[float, list[str]]:
    if not keywords:
        return 0.0, []
    t = (text or "").lower()
    matched = []
    for kw in keywords:
        kw = (kw or "").strip().lower()
        if kw and kw in t:
            matched.append(kw)
    return float(len(set(matched)) / max(1, len(set([k.lower() for k in keywords if k])))), matched


@dataclass
class RerankResult:
    doc: dict
    doc_id: str
    final_score: float
    keep: bool
    reasons: dict[str, Any]


class Reranker:
    """
    Explainable reranking + filtering (no training).

    - rule features: keyword coverage, question-title match boost, length sanity
    - optional LLM relevance scoring (if provided)
    - filter by thresholds to reduce irrelevant context entering LLM
    """

    def __init__(
        self,
        llm_score_json: Optional[Callable[[str], str]] = None,
    ):
        self.llm_score_json = llm_score_json

    def rerank_and_filter(
        self,
        candidates: list[Any],  # Candidate from retriever
        query: str,
        parsed_query: dict,
        top_k: int = 8,
    ) -> tuple[list[RerankResult], dict]:
        intent = (parsed_query.get("intent") or "other").lower()
        keywords = parsed_query.get("keywords", []) or []

        thresholds = self._thresholds_for_intent(intent)

        reranked: list[RerankResult] = []
        for c in candidates:
            q_text = str(c.doc.get("Questions", "") or "")
            a_text = str(c.doc.get("Answers", "") or "")

            cov_q, matched_q = _keyword_coverage(keywords, q_text)
            cov_a, matched_a = _keyword_coverage(keywords, a_text)

            # lightweight lexical match between query tokens and question
            q_tokens = set(_tokenize(query))
            title_tokens = set(_tokenize(q_text))
            token_overlap = (len(q_tokens & title_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0

            fusion = float(c.route_scores.get("_fusion", 0.0))

            # simple, bounded rule score
            rule = 0.0
            rule += 0.55 * cov_q
            rule += 0.25 * cov_a
            rule += 0.20 * token_overlap

            llm_rel = None
            llm_rel_reason = None
            if self.llm_score_json is not None and intent not in ("emergency",):
                # avoid unnecessary cost in crisis mode; rely on strict filters
                try:
                    llm_rel, llm_rel_reason = self._llm_relevance(query, q_text, a_text)
                except Exception:
                    llm_rel, llm_rel_reason = None, None

            # combine: fusion is base retrieval; rule is explainable rerank; optional llm_rel is final gate
            final = (0.75 * fusion) + (0.25 * rule)
            if llm_rel is not None:
                # treat as multiplicative gate to suppress weak docs
                final = float(final * (0.5 + 0.5 * llm_rel))

            keep = True
            if fusion < thresholds["min_fusion"]:
                keep = False
            if max(cov_q, cov_a) < thresholds["min_keyword_coverage"] and token_overlap < thresholds["min_title_overlap"]:
                keep = False

            reranked.append(
                RerankResult(
                    doc=c.doc,
                    doc_id=c.doc_id,
                    final_score=float(final),
                    keep=keep,
                    reasons={
                        "route_scores": dict(c.route_scores),
                        "keyword_coverage": {"q": cov_q, "a": cov_a, "matched_q": matched_q, "matched_a": matched_a},
                        "title_token_overlap": float(token_overlap),
                        "llm_relevance": llm_rel,
                        "llm_relevance_reason": llm_rel_reason,
                    },
                )
            )

        reranked.sort(key=lambda x: x.final_score, reverse=True)

        kept = [r for r in reranked if r.keep]
        kept = kept[:top_k]

        debug = {
            "thresholds": thresholds,
            "kept": len(kept),
            "seen": len(reranked),
        }
        return kept, debug

    @staticmethod
    def _thresholds_for_intent(intent: str) -> dict[str, float]:
        # Emergency: stricter filters to avoid irrelevant advice from wrong context.
        if intent == "emergency":
            return {"min_fusion": 0.35, "min_keyword_coverage": 0.20, "min_title_overlap": 0.15}
        if intent == "definition":
            return {"min_fusion": 0.20, "min_keyword_coverage": 0.15, "min_title_overlap": 0.10}
        if intent == "symptom":
            return {"min_fusion": 0.22, "min_keyword_coverage": 0.15, "min_title_overlap": 0.10}
        return {"min_fusion": 0.18, "min_keyword_coverage": 0.12, "min_title_overlap": 0.08}

    def _llm_relevance(self, query: str, q_text: str, a_text: str) -> tuple[float, str]:
        prompt = f"""
You are a reranker for a mental-health RAG retriever.
Given a user query and a candidate QA pair, output ONLY valid JSON:
{{
  "relevance": number between 0.0 and 1.0,
  "reason": "short reason"
}}

Guidelines:
- Score 1.0 only if the QA directly answers the query.
- Penalize if the QA is about a different topic or too generic.
- Be conservative: prefer lower scores over hallucinating relevance.

Query: {query}
CandidateQuestion: {q_text}
CandidateAnswer: {a_text}
""".strip()
        raw = self.llm_score_json(prompt)
        import json

        obj = json.loads(raw)
        rel = float(obj.get("relevance", 0.0))
        rel = max(0.0, min(1.0, rel))
        reason = str(obj.get("reason", ""))[:300]
        return rel, reason

