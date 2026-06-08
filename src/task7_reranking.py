"""
Task 7 — Reranking Module.

Phương pháp được triển khai:
    1. RRF (Reciprocal Rank Fusion): Gộp kết quả từ nhiều ranker, dùng làm default.
       RRF(d) = sum 1 / (k + rank_r(d)), với k=60 (Cormack et al. 2009).
    2. MMR (Maximal Marginal Relevance): Chọn candidates vừa relevant vừa diverse.
       MMR = lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected_docs))
    3. Cross-encoder reranker (Jina API): Nếu có API key.

Phương pháp mặc định (khi gọi rerank()): Sử dụng RRF kết hợp BM25 keyword scoring
để tái xếp hạng candidates dựa trên mức độ trùng khớp từ khóa với query,
không cần API key hay mô hình nặng.
"""

import math
from typing import Optional


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Tính cosine similarity giữa 2 vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _keyword_score(query: str, content: str) -> float:
    """Tính điểm trùng khớp từ khóa giữa query và content."""
    query_tokens = set(query.lower().split())
    content_tokens = set(content.lower().split())
    if not query_tokens:
        return 0.0
    overlap = query_tokens.intersection(content_tokens)
    return len(overlap) / len(query_tokens)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model (Jina Reranker API).
    Fallback: dùng keyword scoring nếu không có API key.
    """
    # Thử dùng Jina API
    try:
        import os
        import requests
        api_key = os.getenv("JINA_API_KEY", "")
        if api_key and not api_key.startswith("jina_xxx"):
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": top_k
                },
                timeout=10
            )
            if response.status_code == 200:
                reranked = response.json()["results"]
                return [
                    {**candidates[r["index"]], "score": r["relevance_score"]}
                    for r in reranked
                ]
    except Exception:
        pass

    # Fallback: keyword scoring
    scored = []
    for c in candidates:
        kw_score = _keyword_score(query, c["content"])
        # Kết hợp original score và keyword score
        combined = 0.4 * c.get("score", 0) + 0.6 * kw_score
        scored.append({**c, "score": round(combined, 6)})
    
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance -- chọn candidates vừa relevant vừa diverse.
    MMR = lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected_docs))
    """
    if not candidates:
        return []

    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            emb = candidates[idx].get("embedding")
            if not emb:
                relevance = candidates[idx].get("score", 0)
            else:
                relevance = _cosine_similarity(query_embedding, emb)

            # Max similarity to already selected
            max_sim_to_selected = 0
            for sel_idx in selected_indices:
                sel_emb = candidates[sel_idx].get("embedding")
                if emb and sel_emb:
                    sim = _cosine_similarity(emb, sel_emb)
                    max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    results = []
    for i, idx in enumerate(selected_indices):
        item = candidates[idx].copy()
        item["score"] = round(1.0 - i * 0.1, 6) if i < 10 else 0.01
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion -- gộp kết quả từ nhiều ranker.
    RRF(d) = sum 1 / (k + rank_r(d))
    """
    rrf_scores = {}   # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            content_map[key] = item

    # Sort by RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Cau truy van
        candidates: Danh sach candidates tu retrieval
        top_k: So luong ket qua sau rerank
        method: Phuong phap reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Không có query embedding -> fallback sang RRF
        return rerank_rrf([candidates], top_k=top_k)
    elif method == "rrf":
        # Tạo 2 ranked lists: 1 theo original score, 1 theo keyword relevance
        list_by_score = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
        list_by_keyword = sorted(
            candidates,
            key=lambda x: _keyword_score(query, x["content"]),
            reverse=True
        )
        return rerank_rrf([list_by_score, list_by_keyword], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Dieu 248: Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si X bi bat vi su dung ma tuy", "score": 0.7, "metadata": {}},
        {"content": "Hinh phat tu tu 2-7 nam cho toi tang tru", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hinh phat tang tru ma tuy", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
