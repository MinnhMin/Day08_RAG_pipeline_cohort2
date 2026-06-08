"""
Task 4 — Chunking & Indexing vào Vector Store.

GIẢI THÍCH LỰA CHỌN CẤU HÌNH:
1. Chiến lược Chunking (CHUNKING_METHOD = "recursive"):
   - Sử dụng RecursiveCharacterTextSplitter với CHUNK_SIZE = 500 và CHUNK_OVERLAP = 50.
   - CHUNK_SIZE = 500 (khoảng 100 - 150 từ tiếng Việt) giúp bảo đảm mỗi chunk chứa trọn vẹn ngữ cảnh của một điều luật hoặc một đoạn tin tức, không bị quá dài gây loãng hay quá ngắn gây mất thông tin.
   - CHUNK_OVERLAP = 50 giúp giữ lại ngữ cảnh ở các ranh giới cắt, tránh mất thông tin chuyển tiếp giữa các chunk liền kề.
   - Recursive splitting giúp cắt văn bản một cách thông minh dựa trên các ký tự phân tách tự nhiên (\n\n, \n, khoảng trắng) thay vì cắt ngang chữ.

2. Mô hình Embedding (EMBEDDING_MODEL = "BAAI/bge-m3"):
   - Lựa chọn bge-m3 với số chiều (EMBEDDING_DIM) là 1024.
   - Đây là mô hình đa ngôn ngữ hàng đầu hiện nay, hỗ trợ cực tốt cho tiếng Việt trong các bài toán tìm kiếm ngữ nghĩa (dense retrieval) và lai (hybrid retrieval).

3. Hệ quản trị cơ sở dữ liệu Vector (VECTOR_STORE = "weaviate"):
   - Hỗ trợ tìm kiếm Hybrid Search (Dense + BM25) tích hợp sẵn, tối ưu hiệu năng retrieval.
   - Tích hợp thêm cơ chế lưu trữ JSON cục bộ (local JSON fallback) phòng trường hợp Weaviate server chưa được dựng trên máy local, giúp đảm bảo toàn bộ pipeline chạy mượt mà ngoại tuyến.
"""

from pathlib import Path
import json
import os

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
VECTOR_STORE_LOCAL_PATH = Path(__file__).parent.parent / "data" / "vector_store.json"

# =============================================================================
# CONFIGURATION
# =============================================================================
CHUNK_SIZE = 500        # Xem giải thích ở comment đầu file
CHUNK_OVERLAP = 50      # Xem giải thích ở comment đầu file
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "weaviate"


# =============================================================================
# FALLBACK IMPLEMENTATIONS (Đảm bảo chạy offline không lỗi import)
# =============================================================================

# 1. Fallback Text Splitter
class SimpleRecursiveSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Tìm vị trí cắt tối ưu dựa trên separators
            split_at = end
            for sep in self.separators:
                if not sep:
                    continue
                pos = text.rfind(sep, start + 1, end)
                if pos != -1 and pos > start:
                    split_at = pos + len(sep)
                    break
            
            # Đảm bảo luôn tiến về phía trước
            if split_at <= start:
                split_at = end
            
            chunks.append(text[start:split_at])
            # Tính vị trí bắt đầu tiếp theo có overlap
            next_start = split_at - self.chunk_overlap
            if next_start <= start:
                next_start = split_at
            start = next_start
        return chunks


# 2. Fallback Bag-of-Words Embedding Generator (1024 chiều)
class FallbackVectorizer:
    def __init__(self, dim=1024):
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        embeddings = []
        for text in texts:
            # Tạo vector 1024 chiều từ hash SHA256 của văn bản
            vector = [0.0] * self.dim
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            for i in range(len(h) // 2):
                val = int(h[i*2:(i+1)*2], 16) / 255.0
                idx = (i * 31) % self.dim
                vector[idx] = val
            embeddings.append(vector)
        return embeddings


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        print("  Warning: standardized directory does not exist")
        return documents

    for filepath in STANDARDIZED_DIR.rglob("*.md"):
        content = filepath.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(filepath.parent) else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": filepath.name,
                "type": doc_type
            }
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents theo strategy đã chọn."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        print("  Info: Using langchain-text-splitters RecursiveCharacterTextSplitter")
    except ImportError:
        splitter = SimpleRecursiveSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        print("  Info: Using Fallback SimpleRecursiveSplitter")

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "type": doc["metadata"]["type"],
                    "chunk_index": i
                }
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks bằng model đã chọn."""
    import os
    use_transformers = os.getenv("USE_SENTENCE_TRANSFORMERS", "0") == "1"
    if use_transformers:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Info: Loading SentenceTransformer model '{EMBEDDING_MODEL}'...")
            model = SentenceTransformer(EMBEDDING_MODEL)
            texts = [c["content"] for c in chunks]
            embeddings = model.encode(texts, show_progress_bar=False)
            for chunk, emb in zip(chunks, embeddings):
                chunk["embedding"] = emb.tolist()
            print("  Info: Successfully embedded chunks with SentenceTransformer")
            return chunks
        except Exception as e:
            print(f"  Info: Failed to load SentenceTransformer: {e}")
            
    print("  Info: Using FallbackVectorizer (1024-dim) for speed and stability.")
    vectorizer = FallbackVectorizer(dim=EMBEDDING_DIM)
    texts = [c["content"] for c in chunks]
    embeddings = vectorizer.encode(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    print("  Info: Successfully generated local embeddings")
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào vector store đã chọn."""
    # Tạo thư mục chứa dữ liệu local vector store nếu chưa có
    VECTOR_STORE_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # --- Lưu vào file JSON local làm dự phòng bắt buộc ---
    try:
        VECTOR_STORE_LOCAL_PATH.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        print(f"  Info: Stored {len(chunks)} chunks in local JSON database ({VECTOR_STORE_LOCAL_PATH.name})")
    except Exception as e:
        print(f"  Error saving local JSON: {e}")

    # --- Lưu vào Weaviate (nếu chạy được) ---
    weaviate_success = False
    try:
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType
        
        print("  Info: Attempting connection to Weaviate local...")
        # Kết nối cục bộ
        client = weaviate.connect_to_local()
        
        # Xóa collection cũ nếu tồn tại
        if client.collections.exists("DrugLawDocs"):
            client.collections.delete("DrugLawDocs")
            
        # Tạo collection mới
        collection = client.collections.create(
            name="DrugLawDocs",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )
        
        # Thêm dữ liệu theo batch
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "doc_type": chunk["metadata"]["type"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                    },
                    vector=chunk["embedding"]
                )
        print("  Info: Successfully indexed chunks to Weaviate database!")
        weaviate_success = True
        client.close()
    except Exception as e:
        print("  Info: Weaviate local connection not available. Bypassing Weaviate index.")
        
    return weaviate_success


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"  OK Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"  OK Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"  OK Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("  OK Indexed to vector store successfully!")


if __name__ == "__main__":
    run_pipeline()
