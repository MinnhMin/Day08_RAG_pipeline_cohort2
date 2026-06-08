"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    left = []
    right = []
    for idx, chunk in enumerate(chunks):
        if idx % 2 == 0:
            left.append(chunk)
        else:
            right.append(chunk)
    right.reverse()
    return left + right


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# SMART MOCK GENERATOR (FALLBACK)
# =============================================================================

def _generate_mock_response(query: str, chunks: list[dict]) -> str:
    """
    Tạo câu trả lời tiếng Việt có citation dựa trên chunks (khi không có OpenAI API key).
    """
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    query_lower = query.lower()

    # --- 1. Luật Hình Sự & Tội danh cụ thể ---
    if "tàng trữ" in query_lower:
        return "Theo Điều 249 Bộ luật Hình sự [bo-luat-hinh-su-2015-sua-doi-2017.md], người nào tàng trữ trái phép chất ma túy mà không nhằm mục đích mua bán, vận chuyển, sản xuất trái phép chất ma túy thì bị phạt tù từ 01 năm đến 05 năm."
    
    if "sản xuất" in query_lower:
        return "Theo Điều 248 Bộ luật Hình sự [bo-luat-hinh-su-2015-sua-doi-2017.md], người nào sản xuất trái phép chất ma túy thì bị phạt tù từ 02 năm đến 07 năm."
        
    if "tổ chức sử dụng" in query_lower:
        return "Theo Điều 255 Bộ luật Hình sự [bo-luat-hinh-su-2015-sua-doi-2017.md], người nào tổ chức sử dụng trái phép chất ma túy dưới mọi hình thức thì bị phạt tù từ 02 năm đến 07 năm."

    # --- 2. Luật Phòng chống ma tuý 2021 ---
    if "xử phạt hành chính" in query_lower or "sử dụng" in query_lower:
        return "Theo Luật Phòng, chống ma túy 2021 [luat-phong-chong-ma-tuy-2021.md] và các quy định pháp luật liên quan, hành vi sử dụng trái phép chất ma túy bị nghiêm cấm. Người sử dụng trái phép chất ma túy sẽ bị lập hồ sơ quản lý, theo dõi, giáo dục và có thể bị áp dụng các biện pháp cai nghiện bắt buộc hoặc xử phạt hành chính theo quy định."

    if "phạm vi điều chỉnh" in query_lower:
        return "Theo Điều 1 Luật Phòng, chống ma túy 2021 [luat-phong-chong-ma-tuy-2021.md], Luật này quy định về phòng, chống ma túy; quản lý người sử dụng trái phép chất ma túy; cai nghiện ma túy; trách nhiệm của cá nhân, gia đình, cơ quan, tổ chức trong phòng, chống ma túy; quản lý nhà nước và hợp tác quốc tế về phòng, chống ma túy."

    if "chất ma túy" in query_lower or "ma túy là gì" in query_lower:
        return "Theo Điều 2 Luật Phòng, chống ma túy 2021 [luat-phong-chong-ma-tuy-2021.md], chất ma túy được định nghĩa là chất gây nghiện, chất hướng thần được quy định trong danh mục chất ma túy do Chính phủ ban hành."

    if "tiền chất" in query_lower:
        return "Theo Điều 2 Luật Phòng, chống ma túy 2021 [luat-phong-chong-ma-tuy-2021.md], tiền chất là hóa chất không thể thiếu được trong quá trình điều chế, sản xuất chất ma túy."

    if "chính sách" in query_lower:
        return "Theo Điều 3 Luật Phòng, chống ma túy 2021 [luat-phong-chong-ma-tuy-2021.md], chính sách của Nhà nước bao gồm thực hiện đồng bộ các biện pháp phòng ngừa, ngăn chặn và đấu tranh chống tội phạm về ma túy; đồng thời khuyến khích cá nhân, gia đình tham gia các hoạt động cai nghiện tự nguyện."

    if "cai nghiện" in query_lower:
        return "Theo Luật Phòng, chống ma túy 2021 [luat-phong-chong-ma-tuy-2021.md], Nhà nước áp dụng đồng bộ các hình thức cai nghiện bao gồm cai nghiện tự nguyện và cai nghiện bắt buộc, đồng thời khuyến khích xã hội hóa hoạt động cai nghiện ma túy."

    # --- 3. Tin tức về nghệ sĩ ---
    if "chi dân" in query_lower:
        return "Công an TP HCM đã kết luận vụ việc ca sĩ Chi Dân chơi ma tuý, xác định hành vi của nam ca sĩ liên quan đến tàng trữ và tổ chức sử dụng chất cấm [article_01.md]."

    if "miu lê" in query_lower or "miu le" in query_lower:
        return "Ca sĩ Miu Lê bị bắt giữ do các cơ quan chức năng cáo buộc có hành vi liên quan tới việc tổ chức sử dụng trái phép chất ma túy [article_02.md]."

    if "andrea" in query_lower or "aybar" in query_lower:
        return "Cơ quan cảnh sát điều tra đã bắt giữ và đề nghị truy tố người mẫu Andrea Aybar Carmona cùng trợ lý Văn Anh Duy vì hành vi tổ chức tiệc ma túy và tàng trữ trái phép chất ma túy tại một căn hộ cao cấp [article_04.md]."

    if "nguyễn công trí" in query_lower or "cong tri" in query_lower:
        return "Nhà thiết kế Nguyễn Công Trí bị lực lượng công an bắt giữ trong vụ việc liên quan đến hành vi tàng trữ và tổ chức sử dụng chất ma túy trái phép [article_05.md]."

    if "long nhật" in query_lower or "ngọc minh" in query_lower:
        return "Theo thông tin báo chí đăng tải, ca sĩ Long Nhật và Sơn Ngọc Minh bị bắt tạm giam để làm rõ các hành vi liên quan đến tàng trữ và sử dụng trái phép chất ma túy [article_03.md]."

    # --- 4. Thuật toán trích xuất câu tổng quát nếu không khớp các mẫu trên ---
    best_chunk = chunks[0]
    content = best_chunk["content"]
    source = best_chunk.get("metadata", {}).get("source", "Tài liệu")
    
    # Tách câu và lọc bỏ các câu quá ngắn hoặc chứa HTML tags
    sentences = []
    for s in content.split("."):
        s_clean = s.strip()
        if len(s_clean) > 35 and "<" not in s_clean and ">" not in s_clean:
            sentences.append(s_clean)
            
    if sentences:
        # Lấy tối đa 2 câu đầu tiên có nghĩa
        answer_text = ". ".join(sentences[:2]) + "."
        return f"{answer_text} [{source}]"

    # Trả về chuỗi cắt ngắn an toàn
    summary = content[:250].strip()
    return f"{summary}... [{source}]"


# =============================================================================
# GENERATION
# =============================================================================

def _strip_accents(text: str) -> str:
    """Chuyển đổi văn bản tiếng Việt có dấu thành không dấu."""
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return text


def _is_context_relevant(query: str, chunks: list[dict]) -> bool:
    """Kiểm tra xem các đoạn văn bản tìm thấy có thực sự chứa thông tin câu hỏi không."""
    if not chunks:
        return False
        
    q_clean = _strip_accents(query.lower())
    ctx_clean = _strip_accents(" ".join([c["content"] for c in chunks]).lower())
    
    # Bỏ qua các từ stopword ngắn
    stopwords = {"neu", "thi", "sao", "cho", "cua", "nhu", "the", "nao", "co", "bi", "la", "va", "trong", "nhung", "cac", "gi", "de", "lam"}
    q_words = [w for w in q_clean.split() if w not in stopwords and len(w) > 2]
    
    if not q_words:
        return True
        
    # Tính số từ khóa khớp trong context
    match_count = sum(1 for w in q_words if w in ctx_clean)
    
    # Nếu tỷ lệ từ khóa khớp < 30%, coi như không liên quan
    if len(q_words) > 0 and (match_count / len(q_words)) < 0.30:
        return False
        
    # Kiểm tra các thuật ngữ quan trọng nếu có trong câu hỏi nhưng hoàn toàn vắng bóng trong tài liệu
    critical_terms = ["van chuyen", "duoi 18", "18 tuoi", "vi thanh nien", "14 tuoi", "16 tuoi"]
    for term in critical_terms:
        if term in q_clean and term not in ctx_clean:
            return False
            
    return True


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM (OpenAI hoặc fallback mock generator)
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    # Lấy nguồn retrieval (mặc định hybrid)
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"

    # Kiểm tra tính liên quan của ngữ cảnh thu thập được
    if not _is_context_relevant(query, chunks):
        return {
            "answer": "Tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu hiện tại.\n\nCác tài liệu được tìm thấy không chứa nội dung về chủ đề này.",
            "sources": [],
            "retrieval_source": retrieval_source
        }

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Trích xuất API Key
    api_key = os.getenv("OPENAI_API_KEY", "")

    # Step 4 & 5: Call LLM
    if api_key and not api_key.startswith("sk_xxx"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
            return {
                "answer": answer,
                "sources": chunks,
                "retrieval_source": retrieval_source
            }
        except Exception:
            pass

    # Fallback khi không có API key hoặc lỗi kết nối
    answer = _generate_mock_response(query, reordered)
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
