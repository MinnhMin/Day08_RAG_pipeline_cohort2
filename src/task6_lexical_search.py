"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo -> +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document -> điểm cao
    - Inverse Document Frequency (IDF): từ hiếm -> quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = sum IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Cơ chế fallback:
    - Thử dùng rank_bm25 (BM25Okapi) trước.
    - Nếu chưa cài, sử dụng SimpleBM25 tự implement bằng thuần Python
      với cùng công thức BM25 Okapi chuẩn.
"""

import json
import math
from pathlib import Path

VECTOR_STORE_LOCAL_PATH = Path(__file__).parent.parent / "data" / "vector_store.json"

# Cache corpus và BM25 index
_corpus = None
_bm25_index = None


# =============================================================================
# Fallback BM25 Implementation (thuần Python, không cần thư viện ngoài)
# =============================================================================

class SimpleBM25:
    """BM25 Okapi implementation thuần Python."""
    
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.corpus_size = len(corpus_tokens)
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Tính document frequency cho mỗi term
        self.df = {}
        for doc in corpus_tokens:
            seen = set()
            for token in doc:
                if token not in seen:
                    self.df[token] = self.df.get(token, 0) + 1
                    seen.add(token)
        
        # Tính IDF cho mỗi term
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
    
    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for i, doc in enumerate(self.corpus_tokens):
            # Tính term frequency trong document
            tf_map = {}
            for token in doc:
                tf_map[token] = tf_map.get(token, 0) + 1
            
            doc_len = self.doc_lens[i]
            for q_term in query_tokens:
                if q_term not in tf_map:
                    continue
                tf = tf_map[q_term]
                idf = self.idf.get(q_term, 0)
                # BM25 Okapi formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += idf * (numerator / denominator)
        return scores


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def _load_corpus() -> list[dict]:
    """Đọc corpus từ local vector store JSON (tái sử dụng dữ liệu Task 4)."""
    global _corpus
    if _corpus is not None:
        return _corpus
    
    if not VECTOR_STORE_LOCAL_PATH.exists():
        return []
    
    data = json.loads(VECTOR_STORE_LOCAL_PATH.read_text(encoding="utf-8"))
    # Chỉ giữ content và metadata, bỏ embedding để tiết kiệm bộ nhớ
    _corpus = [{"content": c["content"], "metadata": c.get("metadata", {})} for c in data]
    return _corpus


def strip_accents(text: str) -> str:
    """Chuyển đổi văn bản tiếng Việt có dấu thành không dấu."""
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return text


def tokenize_vietnamese(text: str) -> list[str]:
    """Tokenize tiếng Việt giữ lại cả từ gốc và từ không dấu để tối ưu khớp từ khóa."""
    text_lower = text.lower()
    tokens = text_lower.split()
    
    # Tạo tokens không dấu
    unaccented_text = strip_accents(text_lower)
    unaccented_tokens = unaccented_text.split()
    
    # Kết hợp cả hai
    return list(set(tokens + unaccented_tokens))


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # Tokenize sử dụng bộ xử lý tiếng Việt nâng cao
    tokenized_corpus = [tokenize_vietnamese(doc["content"]) for doc in corpus]
    
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(tokenized_corpus)
    except ImportError:
        bm25 = SimpleBM25(tokenized_corpus)
    
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _bm25_index
    
    corpus = _load_corpus()
    if not corpus:
        return []
    
    # Xây dựng index nếu chưa có
    if _bm25_index is None:
        _bm25_index = build_bm25_index(corpus)
    
    # Tokenize query sử dụng bộ xử lý tiếng Việt nâng cao
    tokenized_query = tokenize_vietnamese(query)
    
    # Tính BM25 scores
    scores = _bm25_index.get_scores(tokenized_query)
    
    # Lấy top_k indices có score > 0, sắp xếp giảm dần
    indexed_scores = [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    indexed_scores = indexed_scores[:top_k]
    
    results = []
    for idx, score in indexed_scores:
        results.append({
            "content": corpus[idx]["content"],
            "score": round(float(score), 6),
            "metadata": corpus[idx]["metadata"]
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
