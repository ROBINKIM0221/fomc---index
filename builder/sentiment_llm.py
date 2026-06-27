"""
LLM 에이전트 기반 FOMC 통화정책 스탠스 분석기.

기존 builder/sentiment.py(FinBERT) 의 드롭인 대체 + 연구용 확장.

핵심 차이 (FinBERT 대비):
  - 단순 긍/부정 '감성'이 아니라 통화정책 '스탠스'(매파/비둘기파)를 측정한다.
  - 문장을 독립적으로 보지 않고, 부정/조건/미래시제 맥락을 모델이 추론한다.
  - 점수뿐 아니라 '근거 문장 인용'과 '확신도'를 함께 반환한다(설명가능성).

라벨 규약 (aggregate.py 의 0~100 지수와 호환):
  +1 = dovish  (완화적 / 금리인하·부양 시그널 → 시장에 우호적)
   0 = neutral (중립 / 데이터 의존적, 방향성 없음)
  -1 = hawkish (긴축적 / 금리인상·인플레 경계 시그널)

  → aggregate.py 의 score 평균이 그대로 'dovish-ness 지수'가 된다.
    (FinBERT 의 긍/부정과 부호 방향이 동일: +가 우호적)

환경변수:
  ANTHROPIC_API_KEY   필수. Claude API 키.
  CLAUDE_MODEL        기본 "claude-opus-4-8". 다른 모델로 교체 가능.
  CLAUDE_BATCH        문장 분류 배치 크기(기본 40).
  CLAUDE_EFFORT       추론 강도 low|medium|high (기본 medium).
"""
from __future__ import annotations

import os
from typing import List, Optional

from pydantic import BaseModel, Field

import anthropic

_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")
_BATCH = int(os.getenv("CLAUDE_BATCH", "40"))
_EFFORT = os.getenv("CLAUDE_EFFORT", "medium")

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # ANTHROPIC_API_KEY 를 환경에서 자동으로 읽는다.
        _client = anthropic.Anthropic()
    return _client


# --- 시스템 프롬프트(프롬프트 캐싱 대상: 매 요청마다 동일하게 유지할 것) ---------

_STANCE_SYSTEM = """You are a senior Federal Reserve policy analyst. You read FOMC \
statements and meeting minutes and classify the *monetary-policy stance* expressed \
in each sentence.

Use exactly one label per sentence:
  "dovish"  - signals easing / accommodation / rate cuts / concern about growth or
              employment that argues for looser policy. Market-friendly.
  "hawkish" - signals tightening / rate hikes / inflation vigilance / restraint.
              Market-restrictive.
  "neutral" - procedural, factual, data-dependent, or no directional policy signal.

Rules:
  - Judge the *policy implication*, not surface sentiment. "Inflation remains
    elevated" is hawkish even though it is a plain factual statement.
  - Respect negation, conditionals, and tense. "If inflation does not decline,
    further tightening may be warranted" is hawkish.
  - Boilerplate, voting records, and administrative lines are neutral.
  - Be calibrated: most sentences in FOMC documents are neutral."""


# === 1) 드롭인 대체: 문장 단위 라벨 =======================================

class _SentenceLabel(BaseModel):
    index: int = Field(description="1-based index of the sentence as given")
    stance: str = Field(description="one of: dovish, hawkish, neutral")


class _BatchLabels(BaseModel):
    labels: List[_SentenceLabel]


_STANCE_TO_INT = {"dovish": 1, "neutral": 0, "hawkish": -1}


def predict_labels(sentences: List[str]) -> List[int]:
    """기존 sentiment.py 와 동일한 시그니처. 문장별 스탠스를 -1/0/1 로 반환.

    aggregate.py 가 이 함수를 그대로 호출하므로 코드 수정이 필요 없다.
    """
    if not sentences:
        return []

    client = _get_client()
    out: List[int] = []

    for i in range(0, len(sentences), _BATCH):
        chunk = sentences[i : i + _BATCH]
        numbered = "\n".join(f"{j + 1}. {s}" for j, s in enumerate(chunk))

        resp = client.messages.parse(
            model=_MODEL,
            max_tokens=8000,
            system=[{
                "type": "text",
                "text": _STANCE_SYSTEM,
                "cache_control": {"type": "ephemeral"},  # 동일 시스템 프롬프트 캐싱
            }],
            messages=[{
                "role": "user",
                "content": (
                    "Classify each numbered sentence. Return one label per sentence, "
                    "keeping the same index.\n\n" + numbered
                ),
            }],
            output_format=_BatchLabels,
        )

        parsed = resp.parsed_output
        # index 정렬 보장 (모델이 순서를 바꿔도 위치 복원)
        by_idx = {l.index: _STANCE_TO_INT.get(l.stance.lower(), 0) for l in parsed.labels}
        for j in range(len(chunk)):
            out.append(by_idx.get(j + 1, 0))

    return out


# === 2) 연구 확장: 문서 단위 설명가능 분석 ================================

class PolicySignal(BaseModel):
    quote: str = Field(description="verbatim sentence from the document")
    stance: str = Field(description="dovish | hawkish | neutral")
    reason: str = Field(description="why this signals that stance")


class DocumentAnalysis(BaseModel):
    stance_score: float = Field(
        description="overall stance from -1.0 (fully hawkish) to +1.0 (fully dovish)"
    )
    label: str = Field(description="dovish | hawkish | neutral")
    confidence: float = Field(description="0.0-1.0 confidence in the assessment")
    summary: str = Field(description="2-3 sentence plain-language summary of the stance")
    key_signals: List[PolicySignal] = Field(
        description="the most decisive sentences driving the score, with citations"
    )
    tone_shift: Optional[str] = Field(
        default=None,
        description="how the tone changed vs the previous meeting, if context given",
    )


def analyze_document(text: str, prev_summary: Optional[str] = None) -> DocumentAnalysis:
    """문서 한 건을 설명가능하게 분석한다.

    단순 점수가 아니라 근거 인용·확신도·직전 회의 대비 변화까지 반환한다.
    이것이 FinBERT 대비 연구적 차별점(방향 A)이다.
    """
    client = _get_client()

    user_parts = [
        "Analyze the monetary-policy stance of this FOMC document. "
        "Quote the decisive sentences as evidence and give an overall stance score."
    ]
    if prev_summary:
        user_parts.append(
            "\n\nPrevious meeting summary (for tone-shift comparison):\n" + prev_summary
        )
    user_parts.append("\n\n=== DOCUMENT ===\n" + text)

    resp = client.messages.parse(
        model=_MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},          # 복합 추론 → adaptive thinking
        output_config={"effort": _EFFORT},
        system=[{
            "type": "text",
            "text": _STANCE_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": "".join(user_parts)}],
        output_format=DocumentAnalysis,
    )
    return resp.parsed_output


if __name__ == "__main__":
    # 간단 동작 확인 (ANTHROPIC_API_KEY 필요)
    demo = [
        "Inflation remains elevated and the Committee is strongly committed to returning it to 2 percent.",
        "The labor market has shown signs of cooling in recent months.",
        "Members voted unanimously to maintain the target range.",
        "Should incoming data warrant, the Committee is prepared to ease policy.",
    ]
    print("labels (-1 hawkish / 0 neutral / +1 dovish):")
    print(predict_labels(demo))

    print("\ndocument analysis:")
    a = analyze_document("\n".join(demo))
    print(f"  score={a.stance_score:+.2f}  label={a.label}  conf={a.confidence:.2f}")
    print(f"  summary: {a.summary}")
    for s in a.key_signals:
        print(f"   - [{s.stance}] {s.quote!r} -> {s.reason}")
