# 7-1) 랭그래프 기초 — 단계별 State 흐름 시각화 (스트리밍)
#
# 목적: agent.stream()으로 각 노드가 실행될 때마다
#       공책(State)이 어떻게 쌓이는지 실시간으로 확인
#
# 실행: source /Users/user/00_AI_WORKS/.venv/bin/activate
#       python 7-1_스트리밍.py

# ── 에이전트 구성 (7-1_랭그래프기초.py와 동일) ───────────────
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

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

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int

def llm_call(state: State):
    response = model_with_tools.invoke(
        [SystemMessage(content="당신은 사칙연산을 완벽하게 해내는 유능한 Agent입니다.")]
        + state["messages"]
    )
    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

tools_by_name = {t.name: t for t in tools}

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

agent_builder = StateGraph(State)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent = agent_builder.compile()

# ── 헬퍼 함수 ─────────────────────────────────────────────────
def extract_content(msg):
    c = msg.content
    if isinstance(c, list):
        return c[0].get("text", str(c)) if c else "(비어있음)"
    return c or "(비어있음)"

NODE_LABEL = {
    "llm_call":  "🧠 llm_call  (LLM 작업자)",
    "tool_node": "🔧 tool_node (도구 작업자)",
}

# ── 그래프 구조 출력 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  그래프 구조 (설계도)")
print("=" * 60)
print(agent.get_graph().draw_ascii())

# ── 스트리밍 실행 ─────────────────────────────────────────────
QUESTION = "3과 4를 더한 뒤 7을 곱해줘."

print("=" * 60)
print(f"  질문: {QUESTION}")
print("=" * 60)

accumulated_messages = []   # 공책(State)에 쌓이는 메시지 전체
step = 0

for chunk in agent.stream({"messages": [HumanMessage(content=QUESTION)], "llm_calls": 0}):
    for node_name, node_output in chunk.items():
        step += 1
        new_messages = node_output.get("messages", [])
        accumulated_messages.extend(new_messages)

        label = NODE_LABEL.get(node_name, node_name)
        print(f"\n┌{'─' * 58}┐")
        print(f"│ Step {step}  {label:<46}│")
        print(f"└{'─' * 58}┘")

        # 이번 노드에서 생성된 메시지
        print("  [ 이번 노드 출력 ]")
        for msg in new_messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"    📤 도구 요청 → {tc['name']}({tc['args']})")
            elif hasattr(msg, "tool_call_id"):
                print(f"    ✅ 도구 결과 → {msg.content}")
            else:
                content = extract_content(msg)
                print(f"    💬 최종 답변 → {content}")

        if "llm_calls" in node_output:
            print(f"    📊 LLM 누적 호출: {node_output['llm_calls']}회")

        # 공책(State) 누적 현황
        print(f"\n  [ 공책(State) 현재 상태 — 메시지 {len(accumulated_messages)}개 쌓임 ]")
        for i, msg in enumerate(accumulated_messages):
            msg_type = type(msg).__name__
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                names = [tc["name"] for tc in msg.tool_calls]
                print(f"    [{i+1}] {msg_type}: tool_calls={names}")
            elif hasattr(msg, "tool_call_id"):
                print(f"    [{i+1}] {msg_type}: '{msg.content}'")
            else:
                preview = extract_content(msg)[:50]
                print(f"    [{i+1}] {msg_type}: '{preview}'")

        # 다음 이동 방향 표시
        if node_name == "llm_call":
            last = new_messages[-1] if new_messages else None
            if last and hasattr(last, "tool_calls") and last.tool_calls:
                print(f"\n  → Conditional Edge 판단: tool_calls 있음 ➡  tool_node")
            else:
                print(f"\n  → Conditional Edge 판단: tool_calls 없음 ➡  END")
        elif node_name == "tool_node":
            print(f"\n  → Edge: tool_node ➡  llm_call (고정)")

print("\n" + "=" * 60)
print("  실행 완료")
print("=" * 60)
print(f"  총 {step}단계 / LLM 호출 {step - (step // 2)}회 / 도구 실행 {step // 2}회")
