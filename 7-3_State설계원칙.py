# 7-3) State 설계 원칙
#
# 핵심 두 가지:
#   원칙 1 — 꼭 필요한 것만 저장하라
#   원칙 2 — 가공하지 않은 날것(Raw Data)을 저장하라
#
# 실행: source /Users/user/00_AI_WORKS/.venv/bin/activate
#       python 7-3_State설계원칙.py

from typing import Literal
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

# ══════════════════════════════════════════════════════════════
# 원칙 1: 꼭 필요한 것만 저장하라
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("원칙 1: 꼭 필요한 것만 저장하라")
print("=" * 55)

# ── BAD: 파생 변수를 State에 저장 ────────────────────────────
class BadState(TypedDict):
    email_content: str
    email_length: int      # ← BAD: len(email_content)로 바로 계산 가능
    word_count: int        # ← BAD: 역시 파생 변수

# ── GOOD: 원본만 저장, 파생값은 노드 안에서 계산 ─────────────
class GoodState(TypedDict):
    email_content: str     # ← 원본만 저장

def some_node(state: GoodState):
    email = state["email_content"]
    email_length = len(email)          # 노드 안에서 필요할 때 계산
    word_count = len(email.split())    # 노드 안에서 필요할 때 계산
    # State에 저장하지 않고 로컬에서 사용
    print(f"  [계산] 길이={email_length}, 단어수={word_count} (State에 저장 불필요)")

print("\n[Bad] email_length, word_count를 State에 저장")
print("  → State가 커지고, 동기화 오류 위험 증가")
print("\n[Good] 원본(email_content)만 저장, 파생값은 노드 안에서 계산")
some_node({"email_content": "안녕하세요 환불 요청드립니다 빨리 처리해주세요"})

print("""
저장 O 기준:
  - 영속성: 다음 단계까지 이 정보가 계속 유지되어야 하는가?
  - 비용: 다시 구하려면 API 호출 / DB 검색 비용이 드는가?

저장 X 기준:
  - 파생 변수: 이미 있는 데이터로 바로 계산할 수 있는가?
""")

# ══════════════════════════════════════════════════════════════
# 원칙 2: 가공하지 않은 날것(Raw Data)을 저장하라
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("원칙 2: 날것(Raw Data)을 저장하라")
print("=" * 55)

# ── BAD: 사람이 읽기 좋은 문장으로 가공해서 저장 ─────────────
bad_state_data = {
    "customer_info": "이 고객은 VIP 등급이며 매우 화가 난 상태입니다."
}

def bad_route(state):
    # 라우팅이 불안정 — 문자열 파싱 필요, 프롬프트 변경 시 깨짐
    if "VIP" in state["customer_info"]:   # 문자열 파싱에 의존
        return "vip_service"

# ── GOOD: 딕셔너리(Raw Data)로 저장 ──────────────────────────
good_state_data = {
    "customer_info": {
        "grade": "VIP",
        "status": "Angry"
    }
}

def good_route(state):
    # 라우팅이 안전 — 딕셔너리 키로 정확하게 접근
    if state["customer_info"]["grade"] == "VIP":
        return "vip_service"

print("\n[Bad] 문장으로 가공해서 저장")
print(f"  state['customer_info'] = '{bad_state_data['customer_info']}'")
print("  → 라우팅이 문자열 파싱에 의존 → 프롬프트 변경 시 조건문 깨짐")

print("\n[Good] 날것(딕셔너리)으로 저장")
print(f"  state['customer_info'] = {good_state_data['customer_info']}")
print("  → 라우팅이 딕셔너리 키로 정확하게 동작")
print("  → 각 노드에서 용도에 맞게 프롬프트로 변환해서 사용")

print("""
날것 저장의 두 가지 이유:
  1) 라우팅 안전성: if state["intent"] == "refund" 처럼 깔끔하게 조건문 작성 가능
  2) State와 프롬프트 분리: 동일 데이터를 A노드(요약용), B노드(답장용) 등
     각자 필요한 형태로 변환해서 사용 가능
""")

# ══════════════════════════════════════════════════════════════
# 실전: 이메일 응대 에이전트의 올바른 State 스키마
# ══════════════════════════════════════════════════════════════
print("=" * 55)
print("실전: 이메일 응대 에이전트 State 스키마")
print("=" * 55)

# LLM이 분류해 낼 날것(Raw) 딕셔너리 구조
class EmailClassification(TypedDict):
    intent:   Literal["question", "bug", "billing", "feature", "complex"]
    urgency:  Literal["low", "medium", "high", "critical"]
    topic:    str
    summary:  str

# 에이전트 전체가 공유할 State
class EmailAgentState(TypedDict):
    # [입력 데이터] 다시 복구할 수 없는 원본 — 반드시 저장
    email_content: str
    sender_email:  str

    # [LLM 판단 결과] 조건문에서 쓸 깔끔한 딕셔너리 (프롬프트 문장 아님!)
    classification: EmailClassification | None

    # [비싼 연산 결과] API/DB 호출 결과 — 다시 검색하면 비용 발생
    search_results:   list[str] | None
    customer_history: dict | None

    # [최종 결과물] 루프를 돌며 수정·누적되는 답장과 대화 기록
    draft_response: str | None
    messages:       list[str] | None

print("""
EmailAgentState 설계 포인트:
  - 프롬프트 템플릿이나 포맷된 문자열: 단 하나도 없음
  - 오직 원본 텍스트 / 검색 결과 리스트 / LLM이 추출한 딕셔너리만 존재
  - classification은 Literal 타입으로 허용 값을 명시 → 라우팅 안전성 보장
""")

# 올바르게 초기화된 State 예시
example_state: EmailAgentState = {
    "email_content": "결제 오류가 계속 발생합니다. 빨리 해결해주세요.",
    "sender_email":  "user@example.com",
    "classification": {
        "intent":   "bug",
        "urgency":  "high",
        "topic":    "결제 오류",
        "summary":  "결제 시스템 반복 오류 발생"
    },
    "search_results":   ["결제 오류 FAQ: 캐시 삭제 후 재시도"],
    "customer_history": {"grade": "VIP", "total_orders": 42},
    "draft_response":   None,
    "messages":         None,
}

print("[예시] 올바르게 설계된 State 내용:")
for key, value in example_state.items():
    print(f"  {key}: {value}")

print("\n✅ State는 작고, 명확하고, 날것일수록 좋다.")
print("   → 수십 개 노드의 멀티 에이전트도 디버깅과 확장이 쉬워짐")
