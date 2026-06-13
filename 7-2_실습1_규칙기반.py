# 7-2) 실습 1 — 이메일 응대 에이전트 (규칙 기반 분류)
#
# 설계 전략: 기획 → 노드 → 그래프 → State → 구현
#
# 그래프 구조:
#   START → read_email → classify_intent
#                          ├─(inquiry)──→ search_manual → write_reply → END
#                          └─(complaint)→ escalate_to_human → END
#
# State 특징: add_messages 없이 단순 덮어쓰기(Override)
#   → 단방향 파이프라인이라 과거 기록 불필요
#
# 실행: source /Users/user/00_AI_WORKS/.venv/bin/activate
#       python 7-2_실습1_규칙기반.py

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

model = ChatOllama(model="gemma4:e4b", temperature=0.1)

# ── 1. State 정의 ─────────────────────────────────────────────
# add_messages 없이 선언 → 값이 들어올 때마다 덮어쓰기(Override)
class AgentState(TypedDict):
    email_content: str   # 고객이 보낸 원문 이메일
    category: str        # inquiry(문의) / complaint(불만)
    next_step: str       # 다음 행동 (search_manual / escalate_to_human)
    response: str        # 중간/최종 결과물

# ── 2. Node 정의 ──────────────────────────────────────────────
def read_email(state: AgentState):
    """외부에서 들어온 이메일을 공책에 적어두는 접수처 역할"""
    return {'email_content': state['email_content']}

def classify_intent(state: AgentState):
    """키워드 기반으로 category와 next_step을 결정하는 라우터"""
    print("\n[2] 이메일 내용을 분석합니다...")
    email = state["email_content"]

    if "환불" in email or "빨리" in email:
        category = "complaint"
        next_step = "escalate_to_human"
    else:
        category = "inquiry"
        next_step = "search_manual"

    print(f"  → 분류 결과: {category} / 다음 단계: {next_step}")
    return {"category": category, "next_step": next_step}

def search_manual(state: AgentState):
    """AI 자동 처리 트랙 — 메뉴얼 검색 (write_reply로 이어짐)"""
    print("3-A 진입... 메뉴얼을 검색합니다")

def escalate_to_human(state: AgentState):
    """긴급 이슈 트랙 — 상담원 이관 안내문 생성 후 즉시 종료"""
    print("3-B 진입... 상담원 이관합니다.")
    return {'response': '불편을 드려 죄송합니다. 상담원에게 이관하였으니 잠시 대기해 주시기 바랍니다.'}

def write_reply(state: AgentState):
    """LLM이 이메일 원문을 바탕으로 최종 답변 생성"""
    email = state['email_content']
    system = SystemMessage(content="당신은 친절한 고객 센터 상담 AI입니다. 고객 이메일에 답변해 주세요.")
    response = model.invoke([system, {"role": "user", "content": email}])
    content = response.content
    if isinstance(content, list):
        content = content[0].get('text', str(content)) if content else ""
    return {'response': content}

# ── 3. 라우팅 함수 ────────────────────────────────────────────
def route_email(state: AgentState):
    """State의 next_step 필드를 보고 다음 목적지 결정"""
    return state['next_step']   # 'escalate_to_human' 또는 'search_manual'

# ── 4. 그래프 생성 ────────────────────────────────────────────
agent_builder = StateGraph(AgentState)
agent_builder.add_node("read_email",        read_email)
agent_builder.add_node("classify_intent",   classify_intent)
agent_builder.add_node("search_manual",     search_manual)
agent_builder.add_node("escalate_to_human", escalate_to_human)
agent_builder.add_node("write_reply",       write_reply)

agent_builder.add_edge(START, "read_email")
agent_builder.add_edge("read_email", "classify_intent")
agent_builder.add_conditional_edges(
    "classify_intent",
    route_email,
    ["escalate_to_human", "search_manual"]
)
agent_builder.add_edge("search_manual",     "write_reply")
agent_builder.add_edge("write_reply",       END)
agent_builder.add_edge("escalate_to_human", END)

agent = agent_builder.compile()

# ── 그래프 구조 출력 ──────────────────────────────────────────
print("=" * 55)
print("  그래프 구조")
print("=" * 55)
print(agent.get_graph().draw_ascii())

# ── 5. 테스트 실행 ────────────────────────────────────────────
print("=" * 55)
print("[테스트 1] 단순 문의 → 매뉴얼 검색 트랙")
print("=" * 55)
r1 = agent.invoke({"email_content": "비밀번호 변경 방법을 알려주세요."})
print(f"  category  : {r1['category']}")
print(f"  next_step : {r1['next_step']}")
print(f"  response  : {r1['response'][:80]}...")

print("\n" + "=" * 55)
print("[테스트 2] 불만 이메일 → 상담원 이관 트랙")
print("=" * 55)
r2 = agent.invoke({"email_content": "당장 환불해줘!"})
print(f"  category  : {r2['category']}")
print(f"  next_step : {r2['next_step']}")
print(f"  response  : {r2['response']}")
