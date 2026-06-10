# 7-1) 랭그래프 기초 — 사칙연산 에이전트
# 교재: ChatGoogleGenerativeAI → 로컬 ChatOllama(gemma4:e4b) 대체

# ── 1. Model & Tool 정의 ──────────────────────────────────────
from langchain_ollama import ChatOllama
from langchain.tools import tool

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

# ── 2. State 정의 ─────────────────────────────────────────────
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # add_messages 리듀서: 새 메시지가 덮어쓰지 않고 뒤에 누적(Append)됨
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int  # LLM 호출 횟수 추적용 카운터

# ── 3. Node 정의 ──────────────────────────────────────────────
from langchain_core.messages import SystemMessage, ToolMessage

def llm_call(state: State):
    """LLM이 공책(State)을 보고 답변하거나 도구 사용을 요청하는 작업자"""
    response = model_with_tools.invoke(
        [SystemMessage(content="당신은 사칙연산을 완벽하게 해내는 유능한 Agent입니다.")]
        + state["messages"]
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

tools_by_name = {tool.name: tool for tool in tools}

def tool_node(state: State):
    """LLM의 tool_calls 요청을 받아 실제 함수를 실행하는 작업자"""
    result = []
    last_message = state["messages"][-1]
    for tool_call in last_message.tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        tool_result = tool_fn.invoke(tool_call["args"])
        # tool_call_id: LLM이 발급한 주문 번호 → 반드시 매칭해야 함
        result.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
    return {"messages": result}

# ── 4. 그래프 생성 ────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END

def should_continue(state: State):
    """LLM 응답을 보고 다음 목적지를 결정하는 라우팅 함수 (Conditional Edge)"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"   # 도구 호출 요청이 있으면 → tool_node
    return END               # 없으면 → 종료

agent_builder = StateGraph(State)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")               # 시작 → llm_call
agent_builder.add_edge("tool_node", "llm_call")         # 도구 실행 후 → 다시 llm_call
agent_builder.add_conditional_edges(
    "llm_call",         # 출발지
    should_continue,    # 라우팅 함수
    ["tool_node", END]  # 도착지 목록
)

agent = agent_builder.compile()

# ── 5. 에이전트 실행 ──────────────────────────────────────────
from langchain_core.messages import HumanMessage

def extract_content(message):
    """모델별 content 형식 차이 처리 (OpenAI: 문자열 / Gemini·Ollama: 리스트)"""
    content = message.content
    if isinstance(content, list):
        return content[0].get("text", str(content)) if content else ""
    return content

print("=" * 50)
print("[테스트 1] 단순 질문")
print("=" * 50)
response1 = agent.invoke({"messages": [HumanMessage(content="3과 4를 더해줘")]})
print("응답:", extract_content(response1["messages"][-1]))
print(f"LLM 호출 횟수: {response1['llm_calls']}")

print("\n" + "=" * 50)
print("[테스트 2] 복합 질문 (도구 2번 사용)")
print("=" * 50)
response2 = agent.invoke({"messages": [HumanMessage(content="3과 4를 더한 뒤 7을 곱해줘.")]})
print("응답:", extract_content(response2["messages"][-1]))
print(f"LLM 호출 횟수: {response2['llm_calls']}")

print("\n" + "=" * 50)
print("[State 전체 로그 — 내부 흐름 해부]")
print("=" * 50)
for i, msg in enumerate(response2["messages"]):
    content_preview = str(msg.content)[:60]
    tool_calls_info = ""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls_info = f" → tool_calls: {[tc['name'] for tc in msg.tool_calls]}"
    print(f"  [{i+1}] {type(msg).__name__}{tool_calls_info}: {content_preview}")
