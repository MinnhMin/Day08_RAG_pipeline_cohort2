"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Cơ chế hoạt động:
    1. Đọc local vector store (data/vector_store.json) đã được tạo ở Task 4.
    2. Embed câu query bằng cùng phương pháp với Task 4 (FallbackVectorizer hoặc SentenceTransformer).
    3. Tính cosine similarity giữa query vector và tất cả chunk vectors.
    4. Trả về top_k kết quả có điểm cao nhất, sắp xếp giảm dần.
"""

import json
import math
from pathlib import Path

VECTOR_STORE_LOCAL_PATH = Path(__file__).parent.parent / "data" / "vector_store.json"

# Cache dữ liệu vector store để tránh đọc file nhiều lần
_cached_chunks = None


def _load_vector_store() -> list[dict]:
    """Đọc dữ liệu từ local vector store JSON."""
    global _cached_chunks
    if _cached_chunks is not None:
        return _cached_chunks

    if not VECTOR_STORE_LOCAL_PATH.exists():
        return []

    data = json.loads(VECTOR_STORE_LOCAL_PATH.read_text(encoding="utf-8"))
    _cached_chunks = data
    return data


def _embed_query(query: str) -> list[float]:
    """Embed câu query bằng cùng phương pháp với Task 4."""
    # Thử dùng SentenceTransformer trước
    try:
        from sentence_transformers import SentenceTransformer
        from src.task4_chunking_indexing import EMBEDDING_MODEL
        model = SentenceTransformer(EMBEDDING_MODEL)
        return model.encode(query).tolist()
    except Exception:
        pass

    # Fallback: dùng cùng FallbackVectorizer như Task 4
    try:
        from src.task4_chunking_indexing import FallbackVectorizer, EMBEDDING_DIM
        vectorizer = FallbackVectorizer(dim=EMBEDDING_DIM)
        return vectorizer.encode([query])[0]
    except Exception:
        # Fallback cuối cùng nếu không import được Task 4
        import hashlib
        dim = 1024
        vector = [0.0] * dim
        h = hashlib.sha256(query.encode("utf-8")).hexdigest()
        for i in range(len(h) // 2):
            val = int(h[i*2:(i+1)*2], 16) / 255.0
            idx = (i * 31) % dim
            vector[idx] = val
        return vector


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Tính cosine similarity giữa 2 vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    chunks = _load_vector_store()
    if not chunks:
        return []

    # Embed query
    query_embedding = _embed_query(query)

    # Tính cosine similarity với từng chunk
    scored_results = []
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if not embedding:
            continue
        score = _cosine_similarity(query_embedding, embedding)
        scored_results.append({
            "content": chunk["content"],
            "score": round(score, 6),
            "metadata": chunk.get("metadata", {})
        })

    # Sắp xếp giảm dần theo score
    scored_results.sort(key=lambda x: x["score"], reverse=True)

    # Trả về top_k
    return scored_results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
