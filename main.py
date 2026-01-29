import os
import streamlit as st
from dotenv import load_dotenv
from pdf_summary_service import extract_pdf_text, chat_with_gpt, chat_with_exaone

load_dotenv()

st.set_page_config(page_title="PDF 채팅", page_icon="📄")

# ── Session State 초기화 ──────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── Sidebar ───────────────────────────────
with st.sidebar:
    st.header("설정")

    model_choice = st.radio("모델 선택", ["GPT-4.1-mini", "EXAONE 3.5"])

    if model_choice == "GPT-4.1-mini":
        openai_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPEN_AI_API", ""),
            type="password",
        )
    else:
        exaone_key = st.text_input(
            "EXAONE API Key (Friendli)",
            value=os.getenv("EXAONE_API", ""),
            type="password",
        )
        exaone_model = st.text_input(
            "EXAONE Model ID",
            value=os.getenv("EXAONE_MODEL_ID", ""),
        )

    st.divider()
    uploaded_file = st.file_uploader("PDF 업로드", type=["pdf"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("PDF 텍스트 추출 중..."):
            st.session_state.pdf_text = extract_pdf_text(uploaded_file.read())
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.messages = []

    if st.session_state.pdf_name:
        st.success(f"{st.session_state.pdf_name}")

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# ── Main ──────────────────────────────────
st.title("PDF 채팅")

if not st.session_state.pdf_text:
    st.info("사이드바에서 PDF를 업로드하면 대화를 시작할 수 있습니다.")
    st.stop()

# 기존 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
if prompt := st.chat_input("PDF에 대해 질문하세요"):
    # 키 검증
    if model_choice == "GPT-4.1-mini" and not openai_key:
        st.warning("OpenAI API Key를 입력해주세요.")
        st.stop()
    if model_choice == "EXAONE 3.5" and (not exaone_key or not exaone_model):
        st.warning("EXAONE API Key와 Model ID를 입력해주세요.")
        st.stop()

    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 응답 (스트리밍)
    with st.chat_message("assistant"):
        if model_choice == "GPT-4.1-mini":
            stream = chat_with_gpt(
                st.session_state.pdf_text,
                st.session_state.messages,
                openai_key,
            )
        else:
            stream = chat_with_exaone(
                st.session_state.pdf_text,
                st.session_state.messages,
                exaone_key,
                exaone_model,
            )

        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
