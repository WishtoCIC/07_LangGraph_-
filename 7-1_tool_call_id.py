# 7-1) tool_call_id — 주문 번호(송장 번호) 역할 실습
#
# 핵심 개념:
#   LLM이 여러 도구를 동시에 요청할 때 각 요청에 고유 ID를 부여한다.
#   tool_node는 결과를 돌려줄 때 반드시 그 ID를 함께 명시해야
#   LLM이 "이 결과가 어느 요청의 것인지" 짝을 맞출 수 있다.
#
# 실행: source /Users/user/00_AI_WORKS/.venv/bin/activate
#       python 7-1_tool_call_id.py

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

model = ChatOllama(model="gemma4:e4b", temperature=0.1)

# ── 도구 정의 (날씨 조회 mock) ────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """특정 도시의 현재 날씨를 조회한다."""
    data = {
        "서울": "맑음 🌞 (23°C)",
        "도쿄": "비 🌧️ (18°C)",
        "뉴욕": "흐림 ☁️ (15°C)",
        "런던": "안개 🌫️ (12°C)",
    }
    return data.get(city, f"{city}: 날씨 정보 없음")

tools = [get_weather]
model_with_tools = model.bind_tools(tools)

# ══════════════════════════════════════════════════════════════
# 실습 1: 단일 도구 요청 — tool_call_id 확인
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("실습 1: 단일 도구 요청")
print("=" * 60)

response = model_with_tools.invoke([HumanMessage(content="서울 날씨 알려줘")])

print(f"\n[LLM 응답] tool_calls:")
for tc in response.tool_calls:
    print(f"  이름: {tc['name']}")
    print(f"  인자: {tc['args']}")
    print(f"  ID  : {tc['id']}   ← 고유 주문 번호")

# ── tool_call_id 없이 결과 반환 (잘못된 방법) ──────────────────
print("\n[BAD] tool_call_id 없이 결과만 반환하면?")
bad_result = "맑음"   # ID 없이 결과만
print(f"  결과: '{bad_result}'  → LLM이 어느 요청의 결과인지 알 수 없음 → 에러 또는 혼란")

# ── tool_call_id 매칭 (올바른 방법) ───────────────────────────
print("\n[GOOD] tool_call_id를 매칭해서 반환:")
for tc in response.tool_calls:
    result = get_weather.invoke(tc["args"])
    msg = ToolMessage(content=result, tool_call_id=tc["id"])
    print(f"  ToolMessage(content='{result}', tool_call_id='{tc['id']}')")
    print(f"  → LLM이 ID {tc['id']} 요청의 결과임을 정확히 인식")

# ══════════════════════════════════════════════════════════════
# 실습 2: 동시 다중 도구 요청 — ID 매칭이 왜 필수인지
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("실습 2: 동시 다중 도구 요청 (PDF 예시)")
print("=" * 60)

response2 = model_with_tools.invoke(
    [HumanMessage(content="서울과 도쿄의 날씨를 동시에 알려줘")]
)

print(f"\n[LLM 응답] 도구 {len(response2.tool_calls)}개 동시 요청:")
for i, tc in enumerate(response2.tool_calls):
    print(f"\n  요청 {i+1}:")
    print(f"    name : {tc['name']}")
    print(f"    args : {tc['args']}")
    print(f"    id   : {tc['id']}")

print("\n[tool_node] 각 요청을 실행하고 ID를 짝 맞춰 반환:")
tool_messages = []
for tc in response2.tool_calls:
    result = get_weather.invoke(tc["args"])
    tm = ToolMessage(content=result, tool_call_id=tc["id"])
    tool_messages.append(tm)
    print(f"\n  ToolMessage:")
    print(f"    content      : '{result}'")
    print(f"    tool_call_id : '{tc['id']}'  ← 요청 ID와 정확히 매칭")

print("\n[ID 매칭 시각화]")
for tc, tm in zip(response2.tool_calls, tool_messages):
    city = tc["args"].get("city", "?")
    print(f"  요청 ID {tc['id']} → {city} 요청 → 결과: '{tm.content}' (ID {tm.tool_call_id})")

# ══════════════════════════════════════════════════════════════
# 실습 3: LLM이 ID 매칭된 결과를 받아 최종 답변 생성
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("실습 3: ID 매칭 후 LLM 최종 답변")
print("=" * 60)

messages = [HumanMessage(content="서울과 도쿄의 날씨를 동시에 알려줘")]
messages.append(response2)          # AIMessage (tool_calls 포함)
messages.extend(tool_messages)      # ToolMessage × 2 (ID 매칭됨)

final = model_with_tools.invoke(messages)
content = final.content
if isinstance(content, list):
    content = "".join([b.get("text", "") for b in content if b.get("type") == "text"])

print(f"\n[최종 답변]\n{content}")

print("\n" + "=" * 60)
print("정리: tool_call_id = LLM이 발급한 '주문 번호(송장 번호)'")
print("  요청 → ID 부여 → 도구 실행 → ID 포함해서 반환 → LLM이 짝 맞춤")
print("=" * 60)
