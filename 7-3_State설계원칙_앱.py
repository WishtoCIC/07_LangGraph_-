# 7-3) State 설계 원칙 — 인터랙티브 비교 앱
#
# 실행: source /Users/user/00_AI_WORKS/.venv/bin/activate
#       streamlit run 7-3_State설계원칙_앱.py  (포트 8702)

import streamlit as st
import json
from typing import Literal
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

st.set_page_config(page_title="7-3 State 설계 원칙", page_icon="📋", layout="wide")

model = ChatOllama(model="gemma4:e4b", temperature=0.1)

# ── State 스키마 정의 ──────────────────────────────────────────
class EmailClassification(TypedDict):
    intent:   Literal["question", "bug", "billing", "feature", "complex"]
    urgency:  Literal["low", "medium", "high", "critical"]
    topic:    str
    summary:  str

class EmailAgentState(TypedDict):
    email_content:   str
    sender_email:    str
    classification:  EmailClassification | None
    search_results:  list[str] | None
    customer_history: dict | None
    draft_response:  str | None
    messages:        list[str] | None

# ── LLM 분류 함수 ─────────────────────────────────────────────
def classify_email(email_content: str) -> dict:
    prompt = """고객 이메일을 분석해서 아래 JSON 형식으로만 답하세요. 다른 설명 없이 JSON만 출력하세요.

{
  "intent": "question 또는 bug 또는 billing 또는 feature 또는 complex 중 하나",
  "urgency": "low 또는 medium 또는 high 또는 critical 중 하나",
  "topic": "한 줄 주제",
  "summary": "한 줄 요약"
}"""
    response = model.invoke([SystemMessage(content=prompt), HumanMessage(content=email_content)])
    raw = response.content
    if isinstance(raw, list):
        raw = "".join([b.get("text", "") for b in raw if b.get("type") == "text"])
    raw = raw.strip()
    # JSON 블록 추출
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception:
        return {"intent": "complex", "urgency": "medium", "topic": "파싱 실패", "summary": raw[:50]}

URGENCY_COLOR = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
INTENT_KO = {
    "question": "일반 문의", "bug": "버그/오류",
    "billing": "결제/환불", "feature": "기능 요청", "complex": "복합 이슈"
}

# ── 헤더 ──────────────────────────────────────────────────────
st.title("📋 7-3 State 설계 원칙")
st.caption("두 가지 설계 원칙을 이메일 분류 에이전트로 직접 확인합니다.")

tab1, tab2, tab3 = st.tabs(["원칙 1: 꼭 필요한 것만", "원칙 2: 날것(Raw) 저장", "실전 State 스키마"])

# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("원칙 1: 꼭 필요한 것만 저장하라")
    st.markdown("파생 변수(계산으로 구할 수 있는 값)는 State에 넣지 말고 노드 안에서 계산하세요.")

    email_input1 = st.text_area(
        "이메일 입력",
        value="안녕하세요. 결제가 계속 실패합니다. 빨리 처리해 주세요.",
        height=80, key="tab1_email"
    )

    col_bad, col_good = st.columns(2)

    with col_bad:
        st.markdown("#### ❌ Bad State")
        st.code("""class BadState(TypedDict):
    email_content: str
    email_length: int   # ← 파생 변수
    word_count: int     # ← 파생 변수
    has_urgency: bool   # ← 파생 변수""", language="python")

        if email_input1:
            bad_state = {
                "email_content": email_input1,
                "email_length": len(email_input1),
                "word_count": len(email_input1.split()),
                "has_urgency": any(w in email_input1 for w in ["빨리", "급", "urgent", "즉시"]),
            }
            st.json(bad_state)
            st.warning("State가 커질수록 동기화 오류 위험 증가.\n필드가 많아질수록 관리 비용 증가.")

    with col_good:
        st.markdown("#### ✅ Good State")
        st.code("""class GoodState(TypedDict):
    email_content: str  # ← 원본만 저장

def node(state):
    # 필요할 때 노드 안에서 계산
    length = len(state["email_content"])
    words  = len(state["email_content"].split())""", language="python")

        if email_input1:
            good_state = {"email_content": email_input1}
            st.json(good_state)
            length = len(email_input1)
            words  = len(email_input1.split())
            has_urg = any(w in email_input1 for w in ["빨리", "급", "urgent", "즉시"])
            st.success(f"노드 안에서 필요할 때 계산:\n- 길이: {length}자\n- 단어수: {words}개\n- 긴급 여부: {has_urg}")

    st.info("**저장 O**: 다음 단계까지 유지 필요 / API·DB 재호출 비용 발생\n\n**저장 X**: 이미 있는 데이터로 계산 가능한 파생 변수")

# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("원칙 2: 가공하지 않은 날것(Raw Data)을 저장하라")
    st.markdown("문장(string)이 아니라 딕셔너리(dict) 형태로 저장해야 라우팅이 안전합니다.")

    email_input2 = st.text_area(
        "이메일 입력",
        value="환불 요청드립니다. 3일째 처리가 안 되고 있어요!",
        height=80, key="tab2_email"
    )

    if st.button("▶ LLM으로 분류 후 비교", key="btn2"):
        with st.spinner("LLM 분류 중..."):
            result = classify_email(email_input2)

        col_bad2, col_good2 = st.columns(2)

        with col_bad2:
            st.markdown("#### ❌ Bad State (문장으로 저장)")
            bad_text = (
                f"이 이메일은 {INTENT_KO.get(result.get('intent','?'), '?')} 관련 문의이며 "
                f"긴급도는 {result.get('urgency','?')} 수준입니다. "
                f"주제: {result.get('topic','?')}"
            )
            st.code(f'state["classification"] = "{bad_text}"', language="python")
            st.markdown("**라우팅 시도:**")
            st.code("""# 위험한 라우팅 — 문자열 파싱에 의존
if "환불" in state["classification"]:
    return "billing_track"
# 문장 표현이 조금만 바뀌면 조건문이 깨짐!""", language="python")
            st.error("문장 표현 변경 시 라우팅 오작동 위험")

        with col_good2:
            st.markdown("#### ✅ Good State (날것 딕셔너리 저장)")
            st.code(f'state["classification"] = {json.dumps(result, ensure_ascii=False, indent=2)}', language="python")
            st.markdown("**라우팅:**")
            st.code("""# 안전한 라우팅 — 딕셔너리 키로 접근
if state["classification"]["intent"] == "billing":
    return "billing_track"
if state["classification"]["urgency"] == "critical":
    return "escalate_track"
# 표현이 바뀌어도 키 값은 유지됨""", language="python")

            intent = result.get("intent", "?")
            urgency = result.get("urgency", "?")
            route = "billing_track" if intent == "billing" else ("escalate_track" if urgency == "critical" else "general_track")
            st.success(f"라우팅 결과: **{route}**\n(intent={intent}, urgency={urgency})")

# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("실전: 이메일 응대 에이전트 State 스키마")
    st.markdown("두 원칙이 실전에서 어떻게 녹아있는지 확인하세요.")

    email_input3 = st.text_area(
        "이메일 입력 후 실행하면 State 전체가 채워지는 과정을 확인할 수 있습니다.",
        value="비밀번호 변경 방법을 알려주세요.",
        height=80, key="tab3_email"
    )
    sender = st.text_input("발신자 이메일", value="customer@example.com")

    if st.button("▶ State 채우기 시뮬레이션", key="btn3"):
        state: EmailAgentState = {
            "email_content":    email_input3,
            "sender_email":     sender,
            "classification":   None,
            "search_results":   None,
            "customer_history": None,
            "draft_response":   None,
            "messages":         None,
        }

        col_schema, col_state = st.columns([1, 1])

        with col_schema:
            st.markdown("#### State 스키마 설계 의도")
            st.markdown("""
| 필드 | 저장 이유 |
|---|---|
| `email_content` | 원본 복구 불가 |
| `sender_email` | 원본 복구 불가 |
| `classification` | LLM 재호출 비용 |
| `search_results` | DB 재검색 비용 |
| `customer_history` | DB 재검색 비용 |
| `draft_response` | 루프 누적 결과물 |
| `messages` | 대화 기록 누적 |
            """)

        with col_state:
            st.markdown("#### State 채워지는 과정")

            st.markdown("**① 초기 State** (입력 직후)")
            st.json(state)

            with st.spinner("LLM 분류 중..."):
                classification = classify_email(email_input3)
            state["classification"] = classification

            st.markdown("**② classification 필드 채워짐** (classify_node 실행 후)")
            st.json(state)

            # 가상의 search_results, customer_history 추가
            state["search_results"] = ["비밀번호 변경: 마이페이지 > 보안 설정"]
            state["customer_history"] = {"grade": "일반", "total_orders": 3}

            st.markdown("**③ 검색 결과 추가** (search_node 실행 후)")
            st.json(state)

        urgency = classification.get("urgency", "low")
        intent  = classification.get("intent", "question")
        icon = URGENCY_COLOR.get(urgency, "⚪")
        st.success(
            f"{icon} 분류 완료: **{INTENT_KO.get(intent, intent)}** | "
            f"긴급도: **{urgency}** | 주제: {classification.get('topic', '-')}"
        )
        st.info("💡 State 어디에도 프롬프트 문장이 없습니다. 오직 원본·딕셔너리·리스트만 존재합니다.")
