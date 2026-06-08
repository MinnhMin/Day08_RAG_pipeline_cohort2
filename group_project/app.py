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
import re
import sys
import streamlit as st
from pathlib import Path

# Thêm thư mục gốc của dự án vào sys.path để có thể import từ src
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Cấu hình stdout utf-8 để hiển thị log tiếng Việt ổn định trên Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import các hàm từ RAG Pipeline
from src.task10_generation import generate_with_citation


# =============================================================================
# UTILS & HELPERS
# =============================================================================

def format_citations_to_html(text: str) -> str:
    """Tự động chuyển đổi các trích dẫn dạng [source] thành pill HTML đẹp mắt."""
    pattern = r"\[([^\]]+)\]"
    def replace_pill(match):
        content = match.group(1)
        # Phân loại màu sắc pill dựa trên tên tài liệu
        if "luat" in content.lower() or "bo-luat" in content.lower() or "nghi-dinh" in content.lower():
            bg = "rgba(59, 130, 246, 0.15)"
            color = "#60a5fa"
            border = "rgba(59, 130, 246, 0.3)"
            icon = "⚖️ "
        else:
            bg = "rgba(245, 158, 11, 0.15)"
            color = "#fbbf24"
            border = "rgba(245, 158, 11, 0.3)"
            icon = "📰 "
        return f'<span style="background: {bg}; color: {color}; padding: 2px 8px; border-radius: 6px; font-size: 0.85em; border: 1px solid {border}; font-weight: 600; margin: 0 2px; display: inline-flex; align-items: center; white-space: nowrap;">{icon}{content}</span>'
    
    # Thay thế ký tự xuống dòng thành thẻ br trong HTML để hiển thị đúng xuống dòng
    formatted_text = text.replace("\n", "<br>")
    return re.sub(pattern, replace_pill, formatted_text)


def rewrite_query_with_history(query: str, chat_history: list) -> str:
    """Sử dụng lịch sử trò chuyện để tinh chỉnh câu hỏi kế tiếp (Follow-up) thành Standalone Query."""
    if not chat_history:
        return query
    
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
            
    # Fallback offline
    last_user_msg = next((m["content"] for m in reversed(chat_history) if m["role"] == "user"), "")
    if last_user_msg and len(query.split()) < 4:
        keywords = " ".join([w for w in last_user_msg.split() if len(w) > 3])
        return f"{query} {keywords}"
        
    return query


# =============================================================================
# STREAMLIT CONFIG & PREMIUM STYLING
# =============================================================================

st.set_page_config(
    page_title="RAG DrugLaw Chatbot",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Premium CSS with responsive details, animations and typography
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* Reset & Typography */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: radial-gradient(circle at top center, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }
    
    /* Hide default streamlit layout elements */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {background: transparent;}
    
    /* Header card design */
    .header-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .header-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
        background: linear-gradient(135deg, #60a5fa 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
        margin-bottom: 1rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.2);
        padding: 4px 12px;
        border-radius: 30px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-dot {
        width: 8px; height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        display: inline-block;
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Custom Chat Message Containers */
    .msg-wrapper-user {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1.5rem;
        animation: bubble-in-user 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    .msg-wrapper-assistant {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1.5rem;
        animation: bubble-in-assistant 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    @keyframes bubble-in-user {
        from { opacity: 0; transform: translateY(12px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes bubble-in-assistant {
        from { opacity: 0; transform: translateY(12px) scale(0.98); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    /* Bubble Content styling */
    .bubble-user {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: #ffffff;
        padding: 14px 20px;
        border-radius: 20px 20px 4px 20px;
        max-width: 78%;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.2);
        line-height: 1.55;
        font-size: 0.98rem;
    }
    .bubble-assistant {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #f1f5f9;
        padding: 16px 22px;
        border-radius: 20px 20px 20px 4px;
        max-width: 82%;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.25);
        line-height: 1.6;
        font-size: 0.98rem;
    }

    /* Style for References container in app */
    .sources-container {
        margin-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        padding-top: 10px;
    }
    .source-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        background: rgba(15, 23, 42, 0.8);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# APP LAYOUT
# =============================================================================

# Header Card
st.markdown("""
<div class="header-card">
    <div class="main-title">⚖️ DrugLaw Intelligence</div>
    <div class="sub-title">Trợ lý tra cứu và hỏi đáp thông minh Luật Phòng chống ma tuý & Tin tức</div>
    <div class="status-badge">
        <span class="status-dot"></span>Hệ thống RAG hoạt động tốt
    </div>
</div>
""", unsafe_allow_html=True)

# Khởi tạo bộ nhớ session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "raw_history" not in st.session_state:
    st.session_state.raw_history = []

# Hiển thị lịch sử trò chuyện bằng Custom HTML/CSS
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-wrapper-user">
            <div class="bubble-user">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Assistant
        formatted_html = format_citations_to_html(msg["content"])
        st.markdown(f"""
        <div class="msg-wrapper-assistant">
            <div class="bubble-assistant">{formatted_html}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị references dưới dạng expander
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Tài liệu tham khảo / Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    source_name = src.get("metadata", {}).get("source", f"Source {idx}")
                    doc_type = src.get("metadata", {}).get("type", "Document")
                    # Gán nhãn icon cho loại tài liệu
                    tag_color = "#3b82f6" if doc_type == "legal" else "#f59e0b"
                    tag_name = "Luật Pháp" if doc_type == "legal" else "Tin Tức"
                    
                    st.markdown(f"""
                    <div class="source-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <span style="font-weight: 600; color: #f1f5f9;">[{idx}] {source_name}</span>
                            <span style="background: {tag_color}1a; color: {tag_color}; border: 1px solid {tag_color}33; padding: 2px 8px; border-radius: 20px; font-size: 0.72rem; font-weight: 700;">{tag_name}</span>
                        </div>
                        <div style="font-size: 0.84rem; color: #94a3b8; line-height: 1.5;">{src['content']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Xử lý nhập liệu mới từ User
if prompt := st.chat_input("Hãy đặt câu hỏi về luật ma tuý hoặc tin tức nghệ sĩ..."):
    # 1. Hiển thị câu hỏi của User
    st.markdown(f"""
    <div class="msg-wrapper-user">
        <div class="bubble-user">{prompt}</div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.raw_history.append({"role": "user", "content": prompt})

    # 2. Xử lý logic RAG
    # Tinh chỉnh câu hỏi từ lịch sử hội thoại
    refined_query = rewrite_query_with_history(prompt, st.session_state.raw_history[:-1])
    
    # Nếu câu hỏi được viết lại khác câu hỏi gốc, in thông tin debug phụ
    if refined_query != prompt:
        st.caption(f"🔍 *Hệ thống tự động tinh chỉnh ngữ cảnh hỏi:* **{refined_query}**")
        
    with st.spinner("Đang tra cứu cơ sở dữ liệu pháp luật và tin tức..."):
        # Gọi hàm RAG pipeline hoàn chỉnh (Task 10)
        result = generate_with_citation(refined_query)
        answer = result.get("answer", "Tôi không thể xử lý yêu cầu lúc này.")
        sources = result.get("sources", [])
        retrieval_src = result.get("retrieval_source", "hybrid")

    # Hiển thị bong bóng trả lời với format citation
    formatted_html = format_citations_to_html(answer)
    st.markdown(f"""
    <div class="msg-wrapper-assistant">
        <div class="bubble-assistant">{formatted_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Hiển thị tài liệu tham khảo nếu có
    if sources:
        with st.expander("📚 Tài liệu tham khảo / Sources"):
            for idx, src in enumerate(sources, 1):
                source_name = src.get("metadata", {}).get("source", f"Source {idx}")
                doc_type = src.get("metadata", {}).get("type", "Document")
                tag_color = "#3b82f6" if doc_type == "legal" else "#f59e0b"
                tag_name = "Luật Pháp" if doc_type == "legal" else "Tin Tức"
                
                st.markdown(f"""
                <div class="source-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                        <span style="font-weight: 600; color: #f1f5f9;">[{idx}] {source_name}</span>
                        <span style="background: {tag_color}1a; color: {tag_color}; border: 1px solid {tag_color}33; padding: 2px 8px; border-radius: 20px; font-size: 0.72rem; font-weight: 700;">{tag_name}</span>
                    </div>
                    <div style="font-size: 0.84rem; color: #94a3b8; line-height: 1.5;">{src['content']}</div>
                </div>
                """, unsafe_allow_html=True)
                
    # Lưu vào lịch sử hội thoại
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.session_state.raw_history.append({"role": "assistant", "content": answer})
st.rerun()
