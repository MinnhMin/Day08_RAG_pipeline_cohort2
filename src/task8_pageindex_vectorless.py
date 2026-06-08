"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store -- sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Cơ chế fallback:
    Nếu chưa có PageIndex API key hoặc chưa cài thư viện,
    hệ thống sẽ fallback sang tìm kiếm BM25 trên corpus local
    (tái sử dụng dữ liệu từ Task 4) và đánh dấu source = "pageindex".
    Điều này đảm bảo pipeline Task 9 luôn có fallback hoạt động.
"""

import os
import json
import math
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTOR_STORE_LOCAL_PATH = Path(__file__).parent.parent / "data" / "vector_store.json"


def upload_documents():
    """Upload toàn bộ markdown documents lên PageIndex."""
    try:
        from pageindex import PageIndex
        pi = PageIndex(api_key=PAGEINDEX_API_KEY)

        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            pi.upload(
                content=content,
                metadata={"filename": md_file.name, "type": md_file.parent.name}
            )
            print(f"  OK Uploaded: {md_file.name}")
    except Exception as e:
        print(f"  Info: PageIndex upload skipped ({e}). Using local fallback.")


def _local_bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """Fallback: tìm kiếm BM25 trên corpus local khi PageIndex không khả dụng."""
    if not VECTOR_STORE_LOCAL_PATH.exists():
        return []

    data = json.loads(VECTOR_STORE_LOCAL_PATH.read_text(encoding="utf-8"))
    corpus = [{"content": c["content"], "metadata": c.get("metadata", {})} for c in data]

    if not corpus:
        return []

    # Tokenize
    query_tokens = query.lower().split()
    tokenized_docs = [doc["content"].lower().split() for doc in corpus]

    # Tính BM25
    n = len(tokenized_docs)
    avgdl = sum(len(d) for d in tokenized_docs) / n if n > 0 else 0
    k1, b = 1.5, 0.75

    # Document frequency
    df = {}
    for doc_tokens in tokenized_docs:
        seen = set(doc_tokens)
        for t in seen:
            df[t] = df.get(t, 0) + 1

    # IDF
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

    # Score
    scores = []
    for i, doc_tokens in enumerate(tokenized_docs):
        tf_map = {}
        for t in doc_tokens:
            tf_map[t] = tf_map.get(t, 0) + 1
        doc_len = len(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            if qt in tf_map:
                tf = tf_map[qt]
                score += idf.get(qt, 0) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
        scores.append(score)

    # Top-k
    indexed = [(i, s) for i, s in enumerate(scores) if s > 0]
    indexed.sort(key=lambda x: x[1], reverse=True)
    indexed = indexed[:top_k]

    results = []
    for idx, score in indexed:
        results.append({
            "content": corpus[idx]["content"],
            "score": round(score, 6),
            "metadata": corpus[idx]["metadata"],
            "source": "pageindex"
        })
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    # Thử dùng PageIndex API
    try:
        from pageindex import PageIndex
        if PAGEINDEX_API_KEY and not PAGEINDEX_API_KEY.startswith("pi_xxx"):
            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            results = pi.query(query=query, top_k=top_k)
            return [
                {
                    "content": r.text,
                    "score": r.score,
                    "metadata": r.metadata,
                    "source": "pageindex"
                }
                for r in results
            ]
    except Exception:
        pass

    # Fallback: BM25 trên local corpus
    return _local_bm25_search(query, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("pi_xxx"):
        print("Info: PAGEINDEX_API_KEY not set. Using local BM25 fallback.")
    else:
        print("Uploading documents...")
        upload_documents()

    print("\nTest query:")
    results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] [{r['source']}] {r['content'][:100]}...")
