# 7-1) 랭그래프 기초 — 단계별 실행 흐름 시각화 (Streamlit 앱)
#
# 실행: source /Users/user/00_AI_WORKS/.venv/bin/activate
#       streamlit run 7-1_스트리밍_앱.py

# ── 에이전트 구성 ──────────────────────────────────────────────
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
import streamlit as st

# 에이전트를 세션 상태에 캐시 (매 실행마다 재컴파일 방지)
@st.cache_resource
def build_agent():
    model = ChatOllama(model="gemma4:e4b", temperature=0.1)

    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply a and b."""
        return a * b

    @tool
    def divide(a: int, b: int) -> float:
        """Divide a and b."""
        return a / b

    tools = [add, multiply, divide]
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        llm_calls: int

    def llm_call(state: State):
        response = model_with_tools.invoke(
            [SystemMessage(content="당신은 사칙연산을 완벽하게 해내는 유능한 Agent입니다.")]
            + state["messages"]
        )
        return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

    def tool_node(state: State):
        result = []
        for tc in state["messages"][-1].tool_calls:
            tool_result = tools_by_name[tc["name"]].invoke(tc["args"])
            result.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))
        return {"messages": result}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tool_node"
        return END

    builder = StateGraph(State)
    builder.add_node("llm_call", llm_call)
    builder.add_node("tool_node", tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_edge("tool_node", "llm_call")
    builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    return builder.compile()


def extract_content(msg):
    c = msg.content
    if isinstance(c, list):
        return c[0].get("text", str(c)) if c else "(응답 없음)"
    return c or "(응답 없음)"


NODE_LABEL = {
    "llm_call":  "🧠 llm_call — LLM 작업자",
    "tool_node": "🔧 tool_node — 도구 작업자",
}
NODE_COLOR = {
    "llm_call":  "blue",
    "tool_node": "orange",
}

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph 단계별 실행 시각화",
    page_icon="🔍",
    layout="wide",
)

agent = build_agent()

# ── 헤더 ──────────────────────────────────────────────────────
st.title("🔍 LangGraph 단계별 실행 시각화")
st.caption("agent.stream()으로 공책(State)이 쌓이는 과정을 실시간으로 확인합니다.")

# ── 그래프 구조 ────────────────────────────────────────────────
with st.expander("📊 그래프 구조 (설계도) — 클릭하여 펼치기", expanded=True):
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.code(agent.get_graph().draw_ascii(), language=None)
    with col_b:
        st.markdown("""
**각 노드의 역할**

| 노드 | 역할 |
|---|---|
| `__start__` | 시작점 |
| `llm_call` | LLM에게 판단 요청 |
| `tool_node` | 실제 함수(도구) 실행 |
| `__end__` | 종료 |

**엣지 종류**

- **고정 Edge**: `tool_node` → `llm_call` (항상 이동)
- **Conditional Edge**: `llm_call` → tool_calls 있으면 `tool_node`, 없으면 `END`
        """)

st.divider()

# ── 사이드바: 입력 ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 실행 설정")
    question = st.text_area(
        "질문 입력",
        value="3과 4를 더한 뒤 7을 곱해줘.",
        height=100,
        help="사칙연산이 포함된 질문을 입력하세요.",
    )
    run_btn = st.button("▶ 실행", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("""
**예시 질문**
- `3과 4를 더한 뒤 7을 곱해줘.`
- `10을 2로 나눈 뒤 5를 더해줘.`
- `6 곱하기 8은?`
    """)

# ── 메인: 실행 결과 ────────────────────────────────────────────
if run_btn and question.strip():
    st.subheader(f"💬 질문: {question}")

    accumulated_messages = []
    step = 0
    tool_count = 0
    llm_count = 0

    progress_placeholder = st.empty()

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=question)], "llm_calls": 0}
    ):
        for node_name, node_output in chunk.items():
            step += 1
            new_messages = node_output.get("messages", [])
            accumulated_messages.extend(new_messages)

            if node_name == "llm_call":
                llm_count = node_output.get("llm_calls", llm_count)
            elif node_name == "tool_node":
                tool_count += len(new_messages)

            label = NODE_LABEL.get(node_name, node_name)

            # 각 스텝을 카드 형태로 표시
            with st.container(border=True):
                step_col, badge_col = st.columns([3, 1])
                with step_col:
                    st.markdown(f"### Step {step}  {label}")
                with badge_col:
                    st.markdown(f"<br>", unsafe_allow_html=True)

                left, right = st.columns(2)

                # 왼쪽: 이번 노드 출력
                with left:
                    st.markdown("**📤 이번 노드 출력**")
                    for msg in new_messages:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                st.info(f"도구 요청: `{tc['name']}({tc['args']})`")
                        elif hasattr(msg, "tool_call_id"):
                            st.success(f"도구 결과: **{msg.content}**")
                        else:
                            content = extract_content(msg)
                            if content and content != "(응답 없음)":
                                st.success(f"최종 답변: **{content}**")
                            else:
                                st.caption("(LLM 내부 처리 — 빈 응답)")

                    if node_name == "llm_call":
                        st.caption(f"📊 LLM 누적 호출: {llm_count}회")

                # 오른쪽: 공책(State) 누적 현황
                with right:
                    st.markdown(f"**📓 공책(State) — 메시지 {len(accumulated_messages)}개 쌓임**")
                    for i, msg in enumerate(accumulated_messages):
                        msg_type = type(msg).__name__
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            names = [tc["name"] for tc in msg.tool_calls]
                            st.text(f"[{i+1}] {msg_type}: tool_calls={names}")
                        elif hasattr(msg, "tool_call_id"):
                            st.text(f"[{i+1}] {msg_type}: '{msg.content}'")
                        else:
                            preview = extract_content(msg)[:40]
                            st.text(f"[{i+1}] {msg_type}: '{preview}'")

                # 라우팅 결정 표시
                st.markdown("**→ 다음 이동**")
                if node_name == "llm_call":
                    last = new_messages[-1] if new_messages else None
                    if last and hasattr(last, "tool_calls") and last.tool_calls:
                        st.warning("Conditional Edge: tool_calls 있음 ➡ **tool_node**")
                    else:
                        st.error("Conditional Edge: tool_calls 없음 ➡ **END**")
                elif node_name == "tool_node":
                    st.info("고정 Edge: tool_node ➡ **llm_call**")

    # 최종 요약
    st.divider()
    st.success(f"✅ 실행 완료 — 총 {step}단계 | LLM {llm_count}회 호출 | 도구 {tool_count}회 실행")

    # 최종 답변 (마지막 AIMessage 중 텍스트가 있는 것)
    final_answer = ""
    for msg in reversed(accumulated_messages):
        if type(msg).__name__ == "AIMessage" and not (hasattr(msg, "tool_calls") and msg.tool_calls):
            candidate = extract_content(msg)
            if candidate and candidate != "(응답 없음)":
                final_answer = candidate
                break

    if final_answer:
        st.info(f"**최종 답변:** {final_answer}")

elif run_btn:
    st.warning("질문을 입력해주세요.")
