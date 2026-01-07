import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional


_TOKEN_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[\u4e00-\u9fff]{1,}")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _doc_id(doc: dict) -> str:
    # Dataset has Question_ID; keep robust fallback.
    if "Question_ID" in doc and doc["Question_ID"] is not None:
        return str(doc["Question_ID"])
    return str(hash((doc.get("Questions", ""), doc.get("Answers", ""))))


@dataclass
class Candidate:
    doc: dict
    doc_id: str
    route_scores: dict[str, float]
    route_explanations: dict[str, Any]


class BM25Index:
    """
    Lightweight BM25 implementation (no external deps).
    """

    def __init__(self, docs: list[dict], fields: list[str], k1: float = 1.2, b: float = 0.75):
        self.docs = docs
        self.fields = fields
        self.k1 = k1
        self.b = b

        self.doc_tokens: list[list[str]] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []

        self._build()

    def _build(self) -> None:
        for doc in self.docs:
            text = " ".join(str(doc.get(f, "")) for f in self.fields)
            tokens = _tokenize(text)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0

    def score(self, query_tokens: list[str], doc_idx: int) -> float:
        if not query_tokens:
            return 0.0
        dl = self.doc_lens[doc_idx] or 1
        tf = self.tf[doc_idx]
        score = 0.0
        N = len(self.docs) or 1
        for t in query_tokens:
            f = tf.get(t, 0)
            if f <= 0:
                continue
            n_q = self.df.get(t, 0)
            # idf with +1 to avoid negative values on very frequent terms
            idf = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))
            denom = f + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            score += idf * (f * (self.k1 + 1) / denom)
        return float(score)


class MultiRetriever:
    """
    Multi-route retrieval:
      - tfidf route: via minsearch.Index.search_with_scores
      - bm25 route: lexical BM25
      - keyword route: simple keyword coverage on Questions/Answers

    Returns merged, deduplicated candidates with route-level explanations for interpretability.
    """

    def __init__(self, minsearch_index: Any):
        self.index = minsearch_index
        self.docs = getattr(minsearch_index, "docs", []) or []
        self._bm25 = BM25Index(self.docs, fields=["Questions", "Answers"])

    def _keyword_score(self, keywords: list[str], doc: dict) -> tuple[float, dict]:
        if not keywords:
            return 0.0, {"matched": []}
        q = str(doc.get("Questions", "")).lower()
        a = str(doc.get("Answers", "")).lower()
        matched = []
        score = 0.0
        for kw in keywords:
            kw = (kw or "").strip().lower()
            if not kw:
                continue
            in_q = kw in q
            in_a = kw in a
            if in_q or in_a:
                matched.append(kw)
                score += 2.0 if in_q else 1.0
        # normalize by keyword count to keep within reasonable scale
        norm = max(1.0, float(len(keywords)))
        return float(score / norm), {"matched": matched}

    def retrieve(
        self,
        query: str,
        parsed_query: Optional[dict] = None,
        per_route_k: int = 20,
        boost_dict: Optional[dict] = None,
        intent_weights: Optional[dict[str, float]] = None,
    ) -> list[Candidate]:
        parsed_query = parsed_query or {}
        keywords = parsed_query.get("keywords", []) or []
        intent = (parsed_query.get("intent") or "other").lower()

        boost_dict = boost_dict or {"Questions": 1.0, "Answers": 1.0}
        intent_weights = intent_weights or self.default_intent_weights(intent)

        # 1) tfidf route (existing minsearch)
        tfidf_results = self.index.search_with_scores(
            query=query,
            num_results=per_route_k,
            boost_dict=boost_dict,
        )

        # 2) bm25 route
        q_tokens = _tokenize(query)
        bm25_scores = [(i, self._bm25.score(q_tokens, i)) for i in range(len(self.docs))]
        bm25_scores.sort(key=lambda x: x[1], reverse=True)
        bm25_results = bm25_scores[:per_route_k]

        # 3) keyword coverage route
        kw_results = []
        for i, doc in enumerate(self.docs):
            s, exp = self._keyword_score(keywords, doc)
            if s > 0:
                kw_results.append((i, s, exp))
        kw_results.sort(key=lambda x: x[1], reverse=True)
        kw_results = kw_results[:per_route_k]

        # Merge + dedup by doc_id
        merged: dict[str, Candidate] = {}

        def upsert(doc: dict, route: str, score: float, explanation: dict) -> None:
            did = _doc_id(doc)
            if did not in merged:
                merged[did] = Candidate(doc=doc, doc_id=did, route_scores={}, route_explanations={})
            merged[did].route_scores[route] = float(score)
            merged[did].route_explanations[route] = explanation

        for r in tfidf_results:
            doc = r["doc"]
            upsert(doc, "tfidf", r["score"], {"field_scores": r.get("field_scores", {})})

        for idx, score in bm25_results:
            if score <= 0:
                continue
            doc = self.docs[idx]
            # add token matches for interpretability
            tf = self._bm25.tf[idx]
            matched = [t for t in set(q_tokens) if tf.get(t, 0) > 0][:12]
            upsert(doc, "bm25", score, {"matched_tokens": matched})

        for idx, score, exp in kw_results:
            doc = self.docs[idx]
            upsert(doc, "keyword", score, exp)

        # Compute a fusion score (still expose raw route scores separately)
        candidates = list(merged.values())
        for c in candidates:
            # normalize by max-per-route among merged candidates to reduce scale mismatch
            c.route_scores["_fusion"] = self._fuse(candidates, c, intent_weights=intent_weights)

        candidates.sort(key=lambda c: c.route_scores.get("_fusion", 0.0), reverse=True)
        return candidates

    def _fuse(self, candidates: list[Candidate], c: Candidate, intent_weights: dict[str, float]) -> float:
        routes = ("tfidf", "bm25", "keyword")
        max_by_route = {r: 0.0 for r in routes}
        for x in candidates:
            for r in routes:
                max_by_route[r] = max(max_by_route[r], x.route_scores.get(r, 0.0))

        fused = 0.0
        for r in routes:
            raw = c.route_scores.get(r, 0.0)
            denom = max_by_route[r] or 1.0
            norm = raw / denom
            fused += intent_weights.get(r, 0.0) * norm
        return float(fused)

    @staticmethod
    def default_intent_weights(intent: str) -> dict[str, float]:
        # Intent-driven strategy: definition relies more on lexical precision;
        # advice relies more on semantically similar Q/A patterns.
        intent = (intent or "other").lower()
        if intent == "definition":
            return {"tfidf": 0.35, "bm25": 0.55, "keyword": 0.10}
        if intent == "symptom":
            return {"tfidf": 0.45, "bm25": 0.45, "keyword": 0.10}
        if intent == "emergency":
            # prefer precision; later filter stricter
            return {"tfidf": 0.30, "bm25": 0.60, "keyword": 0.10}
        if intent in ("treatment", "self_help"):
            return {"tfidf": 0.50, "bm25": 0.35, "keyword": 0.15}
        if intent == "advice":
            return {"tfidf": 0.55, "bm25": 0.30, "keyword": 0.15}
        return {"tfidf": 0.45, "bm25": 0.45, "keyword": 0.10}

