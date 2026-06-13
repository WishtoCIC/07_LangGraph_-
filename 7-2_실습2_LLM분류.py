# 7-2) 실습 2 — 이메일 응대 에이전트 (LLM 분류 + Manager-Worker 구조)
#
# 실습 1과의 차이점:
#   - 분류 판단을 if-else(규칙)가 아닌 LLM이 직접 수행
#   - Manager(classify_node) + Worker(consultant_node + tool_node) 분리
#   - consultant_node ↔ tool_node 피드백 루프: 도구 결과를 보고 최종 답변 완성
#   - State에 add_messages 사용: 도구 호출 기록을 보존해야 LLM이 문맥 파악 가능
#
# 그래프 구조:
#   START → classify_node
#               ├─(escalate)──→ escalate_node → END
#               └─(consultant)→ consultant_node
#                                   ├─(tool_calls)→ tool_node ──┐
#                                   │                            │ (루프)
#                                   └────────────────────────────┘
#                                   └─(no tool_calls)→ END
#
# 실행: source /Users/user/00_AI_WORKS/.venv/bin/activate
#       python 7-2_실습2_LLM분류.py

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import (
    AnyMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
)
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

model = ChatOllama(model="gemma4:e4b", temperature=0.1)

# ── 1. Tool, Model 정의 ───────────────────────────────────────
@tool
def search_manual(query: str) -> str:
    """고객의 질문에 답하기 위해 참고할만한 규정이나 매뉴얼을 검색할 때 사용하는 도구."""
    if '비밀번호' in query:
        return '비밀번호 변경은 마이페이지 - 보안 설정에 있음'
    elif '배송' in query:
        return '00택배에서 3일 내 배송 예정임'
    else:
        return '해당 내용 관련 매뉴얼은 찾을 수 없습니다.'

tools = [search_manual]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)   # Worker용 모델 (도구 권한 보유)

# ── 2. State 정의 ─────────────────────────────────────────────
# add_messages 사용: LLM이 "내가 방금 어떤 도구를 호출했지?" 알려면 기록 보존 필수
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    next_step: str   # classify_node가 기록 → route_after_classify가 참조

# ── 3. Node 정의 ──────────────────────────────────────────────
def classify_node(state: AgentState):
    """Manager: LLM이 이메일 의도를 분석해 'consultant' 또는 'escalate' 판단"""
    print("\n--- [1] 분류 단계 (LLM 판단) ---")
    last_message = state["messages"][-1]

    prompt = """당신은 고객 센터 관리자입니다. 고객의 이메일을 분석해서 다음 단계를 결정하세요.

1. 단순 문의나 정보 요청이라면 -> 'consultant' 반환
2. 환불 요청, 불만 제기, 화난 고객이라면 -> 'escalate' 반환

답변은 오직 단어 하나만 하세요."""

    response = model.invoke([SystemMessage(content=prompt), last_message])
    raw = response.content
    if isinstance(raw, list):
        decision = "".join([b.get('text', '') for b in raw if b.get('type') == 'text'])
    else:
        decision = str(raw)

    decision = decision.strip().lower()
    print(f"  → LLM 판단 결과: {decision}")

    if "escalate" in decision:
        return {"next_step": "escalate"}
    return {"next_step": "consultant"}


def consultant_node(state: AgentState):
    """Worker: 도구를 활용하여 고객 질문에 답변 생성 (피드백 루프의 핵심 노드)"""
    print("\n--- [2-A] 상담 AI 답변 생성 중 ---")
    system = SystemMessage(
        content="당신은 고객 센터 상담 AI입니다. "
                "고객 질문에 답할 때 반드시 search_manual 도구를 먼저 호출해 관련 정보를 검색하세요. "
                "검색 결과를 바탕으로 친절하고 명확한 답변을 작성하세요."
    )
    response = model_with_tools.invoke([system] + state['messages'])
    return {'messages': [response]}


def escalate_node(state: AgentState):
    """긴급 이슈: 상담원 이관 안내 메시지를 State에 추가하고 종료"""
    print("\n--- [2-B] 상담원 이관 ---")
    return {'messages': [AIMessage(content='해당 메일은 전문 상담원에게 이관되었습니다.')]}


def tool_node(state: AgentState):
    """도구 실행 노드: consultant_node의 tool_calls 요청을 실제로 수행"""
    print("\n--- [Tool Node] 도구 직접 실행 ---")
    result = []
    last_message = state["messages"][-1]
    for tc in last_message.tool_calls:
        print(f"  → 실행 중: {tc['name']}({tc['args']})")
        tool_result = tools_by_name[tc["name"]].invoke(tc["args"])
        result.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))
    return {"messages": result}

# ── 4. 라우팅 함수 ────────────────────────────────────────────
def route_after_classify(state: AgentState):
    """Manager 판단 결과를 보고 escalate_node 또는 consultant_node로 분기"""
    return state['next_step']   # 'escalate' 또는 'consultant'


def should_continue(state: AgentState):
    """Worker 응답에 tool_calls가 있으면 tool_node로, 없으면 END"""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tool_node"
    return END

# ── 5. 그래프 생성 ────────────────────────────────────────────
agent_builder = StateGraph(AgentState)
agent_builder.add_node("classify_node",   classify_node)
agent_builder.add_node("consultant_node", consultant_node)
agent_builder.add_node("tool_node",       tool_node)
agent_builder.add_node("escalate_node",   escalate_node)

agent_builder.add_edge(START, "classify_node")

agent_builder.add_conditional_edges(
    "classify_node",
    route_after_classify,
    {"escalate": "escalate_node", "consultant": "consultant_node"}
)

agent_builder.add_conditional_edges(
    "consultant_node",
    should_continue,
    ["tool_node", END]
)

# 핵심: 도구 사용이 끝나면 무조건 consultant_node로 복귀 (피드백 루프)
agent_builder.add_edge("tool_node",     "consultant_node")
agent_builder.add_edge("escalate_node", END)

agent = agent_builder.compile()

# ── 그래프 구조 출력 ──────────────────────────────────────────
print("=" * 55)
print("  그래프 구조 (Manager-Worker + 피드백 루프)")
print("=" * 55)
print(agent.get_graph().draw_ascii())

# ── 최종 답변 추출 헬퍼 ──────────────────────────────────────
def extract_final(response):
    last = response["messages"][-1]
    c = last.content
    if isinstance(c, list):
        return "".join([b.get('text', '') for b in c if b.get('type') == 'text'])
    return c

# ── 6. 테스트 실행 ────────────────────────────────────────────
print("=" * 55)
print("[테스트 1] 단순 문의 → consultant 트랙 (도구 사용)")
print("=" * 55)
r1 = agent.invoke({"messages": [HumanMessage(content="비밀번호 변경은 어디서 해?")]})
print(f"\n최종 답변: {extract_final(r1)[:150]}")

print("\n" + "=" * 55)
print("[테스트 2] 불만 이메일 → escalate 트랙")
print("=" * 55)
r2 = agent.invoke({"messages": [HumanMessage(content="당장 환불해줘!")]})
print(f"\n최종 답변: {extract_final(r2)}")
