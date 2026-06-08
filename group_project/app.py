"""
RAG Chatbot Giao Diện Streamlit.

Hỏi đáp về Luật Phòng chống ma tuý và các tin tức liên quan.
Tính năng:
    1. Giao diện Chat trực quan, hiện đại, phong cách Glassmorphism.
    2. Trả lời câu hỏi có kèm Citation (dựa trên Task 10).
    3. Hỗ trợ bộ nhớ hội thoại (Conversation Memory) và tinh chỉnh câu hỏi tiếp theo (Query Rewriting).
    4. Hiển thị chi tiết tài liệu tham khảo (Sources) dưới mỗi câu trả lời.
"""

import os
import sys
import streamlit as st

# Cấu hình stdout utf-8 để hiển thị log tiếng Việt ổn định trên Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import các hàm từ RAG Pipeline
from src.task10_generation import generate_with_citation


# =============================================================================
# CHATBOT MEMORY & QUERY REWRITING
# =============================================================================

def rewrite_query_with_history(query: str, chat_history: list) -> str:
    """
    Sử dụng lịch sử trò chuyện để tinh chỉnh câu hỏi kế tiếp (Follow-up)
    thành câu hỏi độc lập (Standalone Query) trước khi đưa vào retrieval.
    """
    if not chat_history:
        return query
    
    # Chỉ lấy 2 lượt hội thoại gần nhất để tránh loãng ngữ cảnh
    recent_history = chat_history[-2:]
    history_str = ""
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
    
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("sk_xxx"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            prompt = f"""Dựa trên lịch sử hội thoại sau đây:
{history_str}
Hãy viết lại câu hỏi mới nhất của người dùng dưới đây thành một câu hỏi độc lập duy nhất (Standalone Query), đầy đủ ngữ cảnh bằng tiếng Việt để tìm kiếm tài liệu chính xác hơn. Chỉ trả về câu hỏi mới, không giải thích gì thêm.

Câu hỏi mới nhất: {query}
Câu hỏi độc lập:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            rewritten = response.choices[0].message.content.strip()
            if rewritten:
                return rewritten
        except Exception:
            pass
            
    # Fallback offline: Ghép từ khóa từ câu hỏi trước để làm giàu ngữ cảnh tìm kiếm
    last_user_msg = next((m["content"] for m in reversed(chat_history) if m["role"] == "user"), "")
    if last_user_msg and len(query.split()) < 4:
        # Nếu câu hỏi follow-up quá ngắn (vd: "Ở đâu?", "Ai?", "Hình phạt thế nào?")
        # Ghép thêm chủ ngữ từ câu hỏi trước
        keywords = " ".join([w for w in last_user_msg.split() if len(w) > 3])
        return f"{query} {keywords}"
        
    return query


# =============================================================================
# STREAMLIT CONFIG & STYLING (Premium Theme)
# =============================================================================

st.set_page_config(
    page_title="RAG DrugLaw Chatbot",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Premium CSS (Dark Mode & Elegant accents)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
    /* Tổng thể font và background */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Custom Header block */
    .header-container {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
    }
    .main-title {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Định dạng Chat container */
    [data-testid="stChatMessageContainer"] {
        background-color: transparent;
    }
    
    /* Bong bóng chat Assistant */
    .chat-assistant-bubble {
        background-color: #1e293b;
        border-radius: 16px;
        padding: 1.2rem;
        border: 1.5px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        color: #f1f5f9;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* Bong bóng chat User */
    .chat-user-bubble {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.25);
        color: #ffffff;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* Custom Input chat style */
    [data-testid="stChatInput"] {
        border-radius: 30px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: #111827 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Hide default Streamlit footer */
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# APP LAYOUT
# =============================================================================

# Title Block
st.markdown("""
<div class="header-container">
    <div class="main-title">⚖️ DrugLaw RAG Chatbot</div>
    <div class="sub-title">Trợ lý Hỏi đáp Pháp luật Ma tuý & Tin tức liên quan</div>
</div>
""", unsafe_allow_html=True)

# Khởi tạo bộ nhớ session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "raw_history" not in st.session_state:
    st.session_state.raw_history = []  # Lưu để dùng làm context cho query rewriter

# Hiển thị lịch sử trò chuyện
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Assistant bubble
            st.markdown(f'<div class="chat-assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
            # Hiển thị references dưới dạng expander
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Tài liệu tham khảo / Sources"):
                    for idx, src in enumerate(msg["sources"], 1):
                        source_name = src.get("metadata", {}).get("source", f"Source {idx}")
                        doc_type = src.get("metadata", {}).get("type", "Document")
                        st.markdown(f"**[{idx}] {source_name}** *(Loại: {doc_type})*")
                        st.caption(src["content"])
                        st.markdown("---")
        else:
            # User bubble
            st.markdown(f'<div class="chat-user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

# Xử lý nhập liệu mới từ User
if prompt := st.chat_input("Hãy hỏi tôi về luật ma tuý hoặc nghệ sĩ liên quan..."):
    # 1. Hiển thị câu hỏi của User
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-user-bubble">{prompt}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.raw_history.append({"role": "user", "content": prompt})

    # 2. Xử lý logic RAG
    with st.chat_message("assistant"):
        # Tinh chỉnh câu hỏi từ lịch sử hội thoại
        refined_query = rewrite_query_with_history(prompt, st.session_state.raw_history[:-1])
        
        # Nếu câu hỏi được viết lại khác câu hỏi gốc, in thông tin debug ẩn/phụ
        if refined_query != prompt:
            st.caption(f"🔍 *Đang truy vấn với ngữ cảnh:* **{refined_query}**")
            
        with st.spinner("Đang tra cứu cơ sở dữ liệu pháp luật và tin tức..."):
            # Gọi hàm RAG pipeline hoàn chỉnh (Task 10)
            result = generate_with_citation(refined_query)
            answer = result.get("answer", "Tôi không thể xử lý yêu cầu lúc này.")
            sources = result.get("sources", [])
            retrieval_src = result.get("retrieval_source", "hybrid")

        # Hiển thị bong bóng trả lời
        st.markdown(f'<div class="chat-assistant-bubble">{answer}</div>', unsafe_allow_html=True)

        # Hiển thị tài liệu tham khảo nếu có
        if sources:
            with st.expander("📚 Tài liệu tham khảo / Sources"):
                for idx, src in enumerate(sources, 1):
                    source_name = src.get("metadata", {}).get("source", f"Source {idx}")
                    doc_type = src.get("metadata", {}).get("type", "Document")
                    st.markdown(f"**[{idx}] {source_name}** *(Loại: {doc_type})*")
                    st.caption(src["content"])
                    st.markdown("---")
                    
        # Lưu vào lịch sử hội thoại
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
        st.session_state.raw_history.append({"role": "assistant", "content": answer})
