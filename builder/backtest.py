"""
방향 C — 감성/스탠스 지수의 실증 검증 (이벤트 스터디 백테스트).

질문: "우리가 만든 FOMC 지수가 실제 시장 반응을 설명·예측하는가?"

방법:
  - 각 FOMC 발표일을 '이벤트'로 보고, 발표 전후 시장 변화를 측정한다.
      · 10년물 국채금리(^TNX)  : 매파적이면 금리 상승해야 함
      · S&P500(^GSPC)         : 비둘기파적이면 주가 상승해야 함
  - 지수 점수(score)와 시장 변화의 상관/회귀/방향 적중률을 계산한다.

가설(라벨 규약: +가 우호적/비둘기파):
  H1) score ↑ (dovish)  →  10년물 금리 변화 ↓   (음의 상관 기대)
  H2) score ↑ (dovish)  →  S&P500 수익률 ↑      (양의 상관 기대)

데이터:
  - 이벤트 점수: site/data/Statements_date_data.csv (date, score, ...)
  - 시장 데이터: yfinance (API 키 불필요)

의존성: pandas, numpy, yfinance  (scipy 있으면 p-value 정밀 계산)
실행:   py -m builder.backtest
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import SITE_DATA

ROOT = Path(__file__).resolve().parents[1]

# 발표일 기준 이벤트 윈도우: 발표 전 거래일 → 발표 후 거래일
PRE_OFFSET_DAYS = 1   # t-1
POST_OFFSET_DAYS = 1  # t+1


# === 데이터 로드 ==========================================================

def load_event_scores(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """이벤트(=FOMC 발표일)별 지수 점수를 로드한다."""
    path = csv_path or (SITE_DATA / "Statements_date_data.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. 먼저 `py -m builder.build` 로 파이프라인을 돌려 데이터를 생성하세요."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "score"]).sort_values("date").reset_index(drop=True)
    return df[["date", "score"]]


def fetch_market(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """yfinance 로 10년물 금리와 S&P500 종가를 가져온다."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance 가 필요합니다.  pip install yfinance"
        ) from e

    # 윈도우 양쪽으로 여유를 둬서 거래일 누락에 대비
    lo = (start - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    hi = (end + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    data = yf.download(
        ["^TNX", "^GSPC"], start=lo, end=hi,
        progress=False, auto_adjust=True,
    )["Close"]

    data = data.rename(columns={"^TNX": "yield_10y", "^GSPC": "spx"})
    return data.sort_index()


# === 이벤트 반응 계산 =====================================================

def _nearest_on_or_before(idx: pd.DatetimeIndex, day: pd.Timestamp):
    sub = idx[idx <= day]
    return sub[-1] if len(sub) else None


def _nearest_on_or_after(idx: pd.DatetimeIndex, day: pd.Timestamp):
    sub = idx[idx >= day]
    return sub[0] if len(sub) else None


def compute_reactions(events: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """각 이벤트별 발표 전후 시장 변화를 계산한다."""
    idx = market.index
    rows = []
    for _, ev in events.iterrows():
        d = ev["date"].normalize()
        pre_day = _nearest_on_or_before(idx, d - pd.Timedelta(days=PRE_OFFSET_DAYS))
        post_day = _nearest_on_or_after(idx, d + pd.Timedelta(days=POST_OFFSET_DAYS))
        if pre_day is None or post_day is None or pre_day == post_day:
            continue

        y_pre, y_post = market.loc[pre_day, "yield_10y"], market.loc[post_day, "yield_10y"]
        s_pre, s_post = market.loc[pre_day, "spx"], market.loc[post_day, "spx"]
        if any(pd.isna(v) for v in (y_pre, y_post, s_pre, s_post)):
            continue

        rows.append({
            "date": d,
            "score": float(ev["score"]),
            # ^TNX 는 (금리%×10) 으로 표기되므로 ×10 → bp 환산
            "d_yield_bp": (y_post - y_pre) * 10.0,
            "spx_ret_pct": (s_post / s_pre - 1.0) * 100.0,
        })
    return pd.DataFrame(rows)


# === 통계 평가 ============================================================

def _pearson(x: np.ndarray, y: np.ndarray):
    """상관계수와 (가능하면) p-value 반환."""
    if len(x) < 3:
        return float("nan"), float("nan")
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except ImportError:
        r = float(np.corrcoef(x, y)[0, 1])
        # scipy 없을 때 t-근사로 양측 p-value
        n = len(x)
        if abs(r) >= 1.0:
            return r, 0.0
        t = r * math.sqrt((n - 2) / (1 - r * r))
        # 정규근사 (작은 표본에서는 보수적으로 해석)
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        return r, float(p)


def _hit_rate(score: np.ndarray, market_move: np.ndarray, expect_sign: int) -> float:
    """부호가 가설과 일치한 비율. expect_sign: +1 같은 방향 / -1 반대 방향."""
    mask = (score != 0) & (market_move != 0)
    if mask.sum() == 0:
        return float("nan")
    agree = np.sign(score[mask]) * np.sign(market_move[mask]) == expect_sign
    return float(agree.mean())


def evaluate(react: pd.DataFrame) -> dict:
    if react.empty or len(react) < 3:
        return {"n": int(len(react)), "note": "이벤트가 너무 적어 통계 산출 불가"}

    s = react["score"].to_numpy()
    dy = react["d_yield_bp"].to_numpy()
    sp = react["spx_ret_pct"].to_numpy()

    r_y, p_y = _pearson(s, dy)
    r_s, p_s = _pearson(s, sp)

    return {
        "n": int(len(react)),
        "h1_yield": {  # 기대: 음의 상관
            "corr": round(r_y, 3), "p_value": round(p_y, 4),
            "hit_rate": round(_hit_rate(s, dy, expect_sign=-1), 3),
            "as_expected": (r_y < 0),
        },
        "h2_spx": {  # 기대: 양의 상관
            "corr": round(r_s, 3), "p_value": round(p_s, 4),
            "hit_rate": round(_hit_rate(s, sp, expect_sign=+1), 3),
            "as_expected": (r_s > 0),
        },
    }


# === 실행 진입점 ==========================================================

def run_backtest(out_dir: Optional[Path] = None) -> dict:
    out_dir = out_dir or SITE_DATA
    events = load_event_scores()
    market = fetch_market(events["date"].min(), events["date"].max())
    react = compute_reactions(events, market)

    out_dir.mkdir(parents=True, exist_ok=True)
    react.to_csv(out_dir / "backtest_reactions.csv", index=False)

    result = evaluate(react)
    pd.Series(result).to_json(out_dir / "backtest_summary.json", force_ascii=False)
    return result


def _fmt(res: dict) -> str:
    if "h1_yield" not in res:
        return f"n={res.get('n')}  {res.get('note', '')}"
    h1, h2 = res["h1_yield"], res["h2_spx"]
    ok = lambda b: "[O] 가설부합" if b else "[X] 가설반대"
    return (
        f"이벤트 수 n = {res['n']}\n"
        f"  H1 (dovish→금리↓): corr={h1['corr']:+.3f}  p={h1['p_value']:.4f}  "
        f"적중률={h1['hit_rate']:.0%}  {ok(h1['as_expected'])}\n"
        f"  H2 (dovish→주가↑): corr={h2['corr']:+.3f}  p={h2['p_value']:.4f}  "
        f"적중률={h2['hit_rate']:.0%}  {ok(h2['as_expected'])}"
    )


if __name__ == "__main__":
    print("[backtest] FOMC 발표일 이벤트 스터디 실행...")
    res = run_backtest()
    print(_fmt(res))
    print("[backtest] 저장: site/data/backtest_reactions.csv, backtest_summary.json")
