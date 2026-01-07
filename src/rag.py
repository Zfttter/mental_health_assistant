import json
from time import time
try:
    from groq import Groq  # type: ignore
except Exception:  # pragma: no cover
    Groq = None
try:
    from groq import BadRequestError  # type: ignore
except Exception:  # pragma: no cover
    BadRequestError = None

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None
import os
import ingest
import logging
from typing import Optional

from query_parser import parse_query
from retriever import MultiRetriever
from reranker import Reranker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if load_dotenv is not None:
    load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key) if (Groq is not None and groq_api_key) else None

# Load the search index
try:
    index = ingest.load_index()
except Exception as e:
    logger.error(f"Failed to load index: {e}")
    raise

if index is None:
    raise ValueError("Search index could not be loaded")

multi_retriever = MultiRetriever(index)

ENABLE_QUERY_PARSER_LLM = os.getenv("ENABLE_QUERY_PARSER_LLM", "1") == "1"
ENABLE_RERANKER_LLM = os.getenv("ENABLE_RERANKER_LLM", "0") == "1"

reranker = Reranker(llm_score_json=None)  # configured later after llm() exists


def search(query):
    """
    Backwards-compatible single-route retrieval (legacy).
    Prefer retrieve_context() for multi-route + rerank + filtering.
    """
    try:
        return index.search(query=query, num_results=10)
    except Exception as e:
        logger.error(f"Error in search function: {e}")
        return []

prompt_template = """ 
You are a careful mental health assistant. Answer the QUESTION using ONLY the facts in the CONTEXT.

Hard rules:
- If the CONTEXT does not contain enough information to answer, say you don't have enough information in the database and ask one clarifying question.
- Do NOT invent facts, sources, or medical claims not present in the CONTEXT.
- When you use a fact, cite the source_id(s) in plain text like (source_id: 123).
- If the query indicates immediate danger (self-harm/suicide/overdose), prioritize safety guidance: encourage contacting local emergency services or a trusted professional immediately.

CONTEXT:
{context}

QUESTION:
{question}

Answer (plain text):
""".strip()

entry_template = """ 
source_id={Question_ID}
question={Questions}
answer={Answers}
""".strip()

def build_prompt(query, search_results):
    context = ""
    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt, model="mixtral-8x7b-32768"):
    if client is None:
        raise RuntimeError(
            "Groq client is not configured. Install 'groq' and set GROQ_API_KEY to enable LLM calls."
        )
    start_time = time()
    try:
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
    except Exception as e:
        # Groq occasionally decommissions models; retry once with a safe fallback.
        msg = str(e)
        if ("decommissioned" in msg.lower()) or ("model_decommissioned" in msg.lower()):
            fallback_model = os.getenv("FALLBACK_GROQ_MODEL", "llama-3.1-8b-instant")
            if model != fallback_model:
                logger.warning("Model '%s' decommissioned; retrying with '%s'", model, fallback_model)
                response = client.chat.completions.create(
                    model=fallback_model, messages=[{"role": "user", "content": prompt}]
                )
                model = fallback_model
            else:
                raise
        else:
            raise

    answer = response.choices[0].message.content

    token_stats = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    end_time = time()
    response_time = end_time - start_time

    return answer, token_stats, response_time    


def _llm_text(prompt: str, model: str) -> str:
    text, _, _ = llm(prompt, model=model)
    return text


def retrieve_context(query: str, model: str = "mixtral-8x7b-32768") -> tuple[list[dict], dict]:
    """
    Industrial-ish retrieval pipeline:
      Query Understanding -> Multi-retrieval -> Rerank/Filter -> Top-k contexts
    Returns (docs_for_prompt, retrieval_debug)
    """
    # Configure optional LLM helpers (lazy, keeps module deps minimal)
    qp_llm = None
    if ENABLE_QUERY_PARSER_LLM and groq_api_key:
        qp_llm = lambda p: _llm_text(p, model=os.getenv("QUERY_PARSER_MODEL", "llama-3.1-8b-instant"))

    rr_llm = None
    if ENABLE_RERANKER_LLM and groq_api_key:
        rr_llm = lambda p: _llm_text(p, model=os.getenv("RERANKER_MODEL", "llama-3.1-8b-instant"))

    global reranker
    reranker = Reranker(llm_score_json=rr_llm)

    parsed = parse_query(query, llm_json=qp_llm).to_json()

    per_route_k = int(os.getenv("PER_ROUTE_K", "25"))
    top_k = int(os.getenv("CONTEXT_TOP_K", "8"))
    boost_dict = {"Questions": 1.2, "Answers": 1.0}

    candidates = multi_retriever.retrieve(
        query=query,
        parsed_query=parsed,
        per_route_k=per_route_k,
        boost_dict=boost_dict,
    )

    reranked, rr_debug = reranker.rerank_and_filter(
        candidates=candidates,
        query=query,
        parsed_query=parsed,
        top_k=top_k,
    )

    docs = [r.doc for r in reranked]

    retrieval_debug = {
        "parsed_query": parsed,
        "per_route_k": per_route_k,
        "context_top_k": top_k,
        "boost_dict": boost_dict,
        "rerank_debug": rr_debug,
        "top_candidates": [
            {
                "doc_id": r.doc_id,
                "source_id": r.doc.get("Question_ID"),
                "final_score": r.final_score,
                "keep": r.keep,
                "reasons": r.reasons,
            }
            for r in reranked[: min(len(reranked), 8)]
        ],
    }

    return docs, retrieval_debug


def evaluate_relevance(question, answer, model='mixtral-8x7b-32768'):
    eval_prompt = f"""
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:
Question: {question}
Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation as STRICTLY valid JSON (no code blocks, no extra text):

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "brief explanation"
}}
""".strip()

    evaluation, tokens, _ = llm(eval_prompt, model)
    
    try:
        json_eval = json.loads(evaluation)
        relevance = json_eval['Relevance'].upper()  # Ensure it's uppercase
        if relevance not in ["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]:
            logger.warning(f"Unexpected relevance value: {relevance}. Defaulting to PARTLY_RELEVANT.")
            relevance = "PARTLY_RELEVANT"
        return relevance, json_eval['Explanation'], tokens
    except json.JSONDecodeError:
        logger.error(f"Failed to parse evaluation JSON: {evaluation}")
        return "PARTLY_RELEVANT", "Failed to parse evaluation", tokens


def rag(query, model="mixtral-8x7b-32768"):
    t0 = time()

    search_results, retrieval_debug = retrieve_context(query, model=model)
    prompt = build_prompt(query, search_results)
    answer, tokens, response_time = llm(prompt, model=model)

    relevance, explanation, eval_tokens = evaluate_relevance(query, answer, model=model)
    
    t1 = time()
    took = t1 - t0

    answer_data = {
        'answer': answer,
        'model_used': model,
        'response_time': response_time,
        'relevance': relevance,
        'relevance_explanation': explanation,
        'retrieval_debug': retrieval_debug,
        'prompt_tokens': tokens['prompt_tokens'],
        'completion_tokens': tokens['completion_tokens'],
        'total_tokens': tokens['total_tokens'],
        'eval_prompt_tokens': eval_tokens['prompt_tokens'],
        'eval_completion_tokens': eval_tokens['completion_tokens'],
        'eval_total_tokens': eval_tokens['total_tokens'],
    }

    return answer_data