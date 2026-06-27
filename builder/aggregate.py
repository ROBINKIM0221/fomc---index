from __future__ import annotations
import math
import pandas as pd
from pathlib import Path

NUM_COLS = ["score", "positive", "neutral", "negative"]

def _to_index(x) -> int:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 50
    if not math.isfinite(xf):
        return 50
    xf = max(-1.0, min(1.0, xf))
    return int(round((xf + 1.0) * 50.0))

def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True
    )
    if path.suffix == ".json":
        df.to_json(path, orient="records", force_ascii=False, date_format="iso")
    else:
        df.to_csv(path, index=False)

def compute_timeseries(rows: list[dict], out_dir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "sentence"]).copy()

    from .backend import predict_labels  # SENTIMENT_BACKEND 로 finbert/llm 토글
    df["label"] = predict_labels(df["sentence"].astype(str).tolist())

    df["positive"] = (df["label"] == 1).astype(float)
    df["neutral"]  = (df["label"] == 0).astype(float)
    df["negative"] = (df["label"] == -1).astype(float)

    daily = (
        df.groupby([pd.Grouper(key="date", freq="D"), "doc_type"], as_index=False)
          .agg(score=("label", "mean"),
               positive=("positive", "mean"),
               neutral=("neutral", "mean"),
               negative=("negative", "mean"))
          .sort_values("date")
    )

    for c in NUM_COLS:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")

    def resample(doc_type: str):
        part = daily[daily["doc_type"] == doc_type].copy()
        part_num = part[["date"] + NUM_COLS].set_index("date").sort_index()
        d = part_num.reset_index()
        m = part_num.resample("MS").mean(numeric_only=True).reset_index()
        q = part_num.resample("QS").mean(numeric_only=True).reset_index()
        return d, m, q

    st_daily, st_month, st_quarter = resample("statement")
    mn_daily, mn_month, mn_quarter = resample("minutes")

    headline_m = (
        pd.concat([st_month.assign(src="statement"),
                   mn_month.assign(src="minutes")], ignore_index=True)
          .groupby("date", as_index=False)["score"].mean()
    )
    headline_m["score"] = pd.to_numeric(headline_m["score"], errors="coerce").fillna(0.0)
    headline_m["index_0_100"] = headline_m["score"].apply(_to_index)

    headline_q = (
        pd.concat([st_quarter.assign(src="statement"),
                   mn_quarter.assign(src="minutes")], ignore_index=True)
          .groupby("date", as_index=False)["score"].mean()
    )
    headline_q["score"] = pd.to_numeric(headline_q["score"], errors="coerce").fillna(0.0)
    headline_q["index_0_100"] = headline_q["score"].apply(_to_index)

    _save(st_daily.rename(columns={"date": "date"}), out_dir / "Statements_date_data.csv")
    _save(st_month[["date","score"]], out_dir / "Statements_month_data.csv")
    _save(st_quarter[["date","score"]], out_dir / "Statements_quarter_data.csv")
    _save(st_month[["date","positive","neutral","negative"]], out_dir / "Statements_month_ratio.csv")
    _save(st_quarter[["date","positive","neutral","negative"]], out_dir / "Statements_quarter_ratio.csv")

    _save(mn_daily.rename(columns={"date": "date"}), out_dir / "Minutes_date_data.csv")
    _save(mn_month[["date","score"]], out_dir / "Minutes_month_data.csv")
    _save(mn_quarter[["date","score"]], out_dir / "Minutes_quarter_data.csv")
    _save(mn_month[["date","positive","neutral","negative"]], out_dir / "Minutes_month_ratio.csv")
    _save(mn_quarter[["date","positive","neutral","negative"]], out_dir / "Minutes_quarter_ratio.csv")

    _save(headline_m[["date","score","index_0_100"]], out_dir / "index_monthly.json")
    _save(headline_q[["date","score","index_0_100"]], out_dir / "index_quarterly.json")

    if not headline_m.empty:
        latest = headline_m.sort_values("date").tail(1)
        latest.to_json(out_dir / "latest_monthly.json",
                       orient="records", force_ascii=False, date_format="iso")
