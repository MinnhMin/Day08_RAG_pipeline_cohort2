# RAG Evaluation Results

## Framework sử dụng

> **DeepEval (Real API Mode)**

---

## Overall Scores

| Metric | Config A (Hybrid + Rerank) | Config B (Dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.77 | 0.75 | +0.02 |
| Answer Relevance | 0.93 | 0.89 | +0.04 |
| Context Recall | 0.80 | 0.59 | +0.21 |
| Context Precision | 0.85 | 0.77 | +0.08 |
| **Average** | **0.84** | **0.75** | **+0.09** |

---

## A/B Comparison Analysis

**Config A (Hybrid + Rerank):**
* Sử dụng kết hợp giữa Semantic Search (Dense Retrieval) và BM25 Lexical Search (Sparse Retrieval).
* Kết quả từ hai bộ tìm kiếm được ghép nối bằng giải thuật RRF (Reciprocal Rank Fusion) và áp dụng cấu hình Reranking.
* Hỗ trợ tự động fallback sang PageIndex Vectorless RAG khi kết quả tìm kiếm có điểm số quá thấp.

**Config B (Dense-only):**
* Chỉ sử dụng duy nhất mô hình Semantic Search để tìm kiếm các văn bản liên quan dựa trên Cosine Similarity, không áp dụng thêm bất kỳ bộ lọc từ khóa hoặc reranking nào.

**Kết luận:**
* Cấu hình **Config A (Hybrid + Rerank)** đạt điểm số trung bình vượt trội hơn hẳn Config B (+0.09). 
* Điểm số cải thiện rõ rệt nhất ở chỉ số **Context Recall** và **Context Precision** nhờ vào sự kết hợp giữa tìm kiếm ngữ nghĩa và tìm kiếm từ khóa chính xác của BM25, giúp bao quát đầy đủ thông tin pháp luật có cấu trúc chặt chẽ.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Luật Phòng chống ma tuý 2021 quy định những hình thức cai nghiện nào? | 0.54 | 0.83 | 0.61 | Retrieval | Từ khóa tìm kiếm quá trừu tượng hoặc tài liệu nguồn chưa được chunking hợp lý. |
| 2 | Trách nhiệm cai nghiện ma túy tự nguyện được khuyến khích cho ai theo Luật 2021? | 0.56 | 0.83 | 0.61 | Retrieval | Từ khóa tìm kiếm quá trừu tượng hoặc tài liệu nguồn chưa được chunking hợp lý. |
| 3 | Chính sách của Nhà nước về phòng, chống ma túy gồm những gì theo Điều 3? | 0.62 | 1.00 | 0.61 | Retrieval | Từ khóa tìm kiếm quá trừu tượng hoặc tài liệu nguồn chưa được chunking hợp lý. |

---

## Recommendations

### Cải tiến 1: Tối ưu hóa Chunking Strategy
* **Action:** Sử dụng MarkdownHeaderTextSplitter thay cho RecursiveCharacterTextSplitter đối với tài liệu Luật để giữ nguyên cấu trúc phân cấp Điều/Khoản.
* **Expected impact:** Tăng điểm Context Precision và giảm nhiễu ngữ cảnh đầu vào cho LLM.

### Cải tiến 2: Fine-tune Reranker Model
* **Action:** Tích hợp trực tiếp Cohere Reranker hoặc BAAI/bge-reranker-large local thay vì chỉ sử dụng RRF từ khóa.
* **Expected impact:** Cải thiện vị trí của các thông tin quan trọng nhất trong context, nâng cao chất lượng câu trả lời.

### Cải tiến 3: Mở rộng bộ dữ liệu tin tức viết tắt
* **Action:** Bổ sung từ điển từ đồng nghĩa viết tắt (ví dụ: 'ma túy', 'chất cấm', 'MDMA', 'kẹo') vào tầng Lexical Search.
* **Expected impact:** Tăng cường độ chính xác tìm kiếm (Context Recall) đối với các truy vấn sử dụng thuật ngữ tiếng lóng.
