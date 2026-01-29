import fitz
import tempfile
import os
from openai import OpenAI

SYSTEM_PROMPT_TEMPLATE = (
    "당신은 PDF 문서 전문 어시스턴트입니다. "
    "아래 PDF에서 추출한 텍스트를 바탕으로 사용자의 질문에 한국어로 답변하세요. "
    "문서에 없는 내용은 추측하지 마세요.\n\n"
    "=== PDF 내용 ===\n{pdf_text}\n=== PDF 끝 ==="
)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """PDF 전체 텍스트를 추출합니다."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        pages = [doc[i].get_text() for i in range(len(doc))]
        doc.close()
        return "\n\n".join(pages).strip()
    finally:
        os.unlink(tmp_path)


def chat_with_gpt(pdf_text: str, messages: list[dict], api_key: str):
    """GPT-4.1-mini 스트리밍 대화."""
    from openai import APIError, APIConnectionError, APIStatusError

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(pdf_text=pdf_text)},
                *messages,
            ],
            temperature=0.3,
            stream=True,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except APIConnectionError:
        yield "⚠️ OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인해주세요."
    except APIStatusError as e:
        if e.status_code == 401:
            yield "⚠️ API 키가 유효하지 않습니다."
        elif e.status_code == 429:
            yield "⚠️ API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
        else:
            yield f"⚠️ API 오류 ({e.status_code}): {e.message}"
    except APIError as e:
        yield f"⚠️ API 오류: {e.message}"


def chat_with_exaone(pdf_text: str, messages: list[dict], api_key: str, model_id: str):
    """EXAONE(Friendli AI) 스트리밍 대화."""
    from openai import APIError, APIConnectionError, APIStatusError

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.friendli.ai/dedicated/v1",
    )

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(pdf_text=pdf_text)},
                *messages,
            ],
            temperature=0.7,
            max_tokens=512,
            stream=True,
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except APIConnectionError:
        yield "⚠️ EXAONE 서버에 연결할 수 없습니다. 네트워크 상태를 확인해주세요."
    except APIStatusError as e:
        if e.status_code == 500:
            yield "⚠️ EXAONE 엔드포인트가 비활성화 상태입니다. Friendli AI 콘솔에서 엔드포인트를 깨워주세요."
        elif e.status_code == 401:
            yield "⚠️ API 키가 유효하지 않습니다."
        elif e.status_code == 404:
            yield "⚠️ Model ID가 존재하지 않습니다."
        else:
            yield f"⚠️ API 오류 ({e.status_code}): {e.message}"
    except APIError as e:
        yield f"⚠️ API 오류: {e.message}"
