import os
from datetime import datetime
from pathlib import Path
from .utils import SITE_DATA
from .scrape import discover_statements, discover_minutes_from_calendar, download_and_extract
from .preprocess import preprocess_text_files
from .aggregate import compute_timeseries

def run_online():
    from .backend import active_backend
    print("[info] sentiment backend =", active_backend())

    statements = discover_statements()
    minutes = discover_minutes_from_calendar()

    # 최근 N개만 추론(속도/요금 절충). KEEP_N이 없으면 전체.
    keep_n_env = os.getenv("KEEP_N")
    if keep_n_env:
        try:
            n = int(keep_n_env)
            if n > 0:
                statements = statements[-n:]
                minutes = minutes[-n:]
        except ValueError:
            pass

    text_paths: list[str] = []

    for d, url in (statements + minutes):
        try:
            t = download_and_extract(url, date_hint=str(d))
            text_paths.append(t)
        except Exception as e:
            print("[warn] Download failed", url, e)

    # 수동 txt도 병합
    manual_dir = Path(__file__).resolve().parents[1] / "raw-data" / "manual"
    if manual_dir.exists():
        text_paths += [str(p) for p in manual_dir.glob("*.txt")]

    rows = preprocess_text_files(text_paths)
    compute_timeseries(rows, SITE_DATA)
    print("[OK] Build completed at", datetime.utcnow().isoformat() + "Z")

if __name__ == "__main__":
    run_online()
