"""
감성/스탠스 분석 백엔드 선택기.

환경변수 SENTIMENT_BACKEND 로 분류기를 토글한다.
  "finbert" (기본) -> builder/sentiment.py     (ProsusAI/finbert, 로컬 추론)
  "llm"            -> builder/sentiment_llm.py (Claude 에이전트, API 키 필요)

aggregate.py 는 이 모듈의 predict_labels 만 호출하므로,
빌드 시 환경변수만 바꾸면 코드 수정 없이 백엔드가 교체된다.

  예) SENTIMENT_BACKEND=llm CLAUDE_MODEL=claude-opus-4-8 py -m builder.build

두 백엔드 모두 동일 규약을 따른다: 문장 리스트 -> [-1, 0, 1] 라벨 리스트.
부호 방향도 호환된다(+ = 우호적/비둘기파).
"""
from __future__ import annotations

import os
from typing import List

_VALID = {"finbert", "llm"}


def _resolve() -> str:
    backend = os.getenv("SENTIMENT_BACKEND", "finbert").strip().lower()
    if backend not in _VALID:
        raise ValueError(
            f"SENTIMENT_BACKEND='{backend}' 는 지원하지 않음. {sorted(_VALID)} 중 선택."
        )
    return backend


def active_backend() -> str:
    """현재 선택된 백엔드 이름."""
    return _resolve()


def predict_labels(sentences: List[str]) -> List[int]:
    """선택된 백엔드로 문장별 라벨(-1/0/1)을 예측한다."""
    backend = _resolve()
    if backend == "llm":
        from .sentiment_llm import predict_labels as _fn
    else:
        from .sentiment import predict_labels as _fn
    return _fn(sentences)
