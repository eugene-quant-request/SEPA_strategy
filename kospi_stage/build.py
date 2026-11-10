# -*- coding: utf-8 -*-
"""data/kospi_daily.csv + stage.js + template.html → kospi_stage_dashboard.html (단일 파일, 오프라인 동작)
    python fetch_kospi.py && python build.py
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    df = pd.read_csv(ROOT / "data" / "kospi_daily.csv", parse_dates=["date"])
    now = datetime.now()
    last = df["date"].iloc[-1].date()
    data = {
        "D": {"d": df["date"].dt.strftime("%Y-%m-%d").tolist(),
              **{k[0]: df[col].round(2).tolist() for k, col in
                 [("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")]}},
        "provisional": bool(last == now.date() and now.hour < 18),
        "built": now.strftime("%Y-%m-%d %H:%M"),
    }
    html = (ROOT / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__STAGE_JS__*/", (ROOT / "stage.js").read_text(encoding="utf-8"))
    html = html.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    out = ROOT / "kospi_stage_dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"{out.name}: {len(df)}일, 기준일 {last}{' (잠정)' if data['provisional'] else ''}, {out.stat().st_size/1e6:.2f}MB")


if __name__ == "__main__":
    main()
