# -*- coding: utf-8 -*-
"""코스피 일봉(OHLCV)을 한투 업종 기간별시세(FHKUP03500100)로 받아 data/kospi_daily.csv 에 이어 붙인다.

- 1회 호출에 50건까지만 오므로 끝 날짜를 당겨 가며 반복한다(최초 실행 ~90회, 이후 1~2회).
- 한투 당일 종가는 시간외 단일가(16~18시)까지 바뀐다. 그래서 매 실행마다 최근 7일은 다시 받아 덮어쓴다.
    python fetch_kospi.py            # 증분
    python fetch_kospi.py --full     # 처음부터 다시
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "kis_api"))
from kis_client import KISClient  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "kospi_daily.csv"
START = pd.Timestamp("2009-01-01")
CODE = "0001"          # KOSPI


def fetch_range(c: KISClient, a: pd.Timestamp, z: pd.Timestamp) -> pd.DataFrame:
    rows, end = [], z
    while end >= a:
        part = c.index_daily_chart(CODE, a.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        part = [r for r in part if r.get("stck_bsop_date")]
        if not part:
            break
        rows += part
        first = min(pd.Timestamp(r["stck_bsop_date"]) for r in part)
        end = first - pd.Timedelta(days=1)
        time.sleep(0.08)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame({
        "date": pd.to_datetime([r["stck_bsop_date"] for r in rows]),
        "open": [float(r["bstp_nmix_oprc"]) for r in rows],
        "high": [float(r["bstp_nmix_hgpr"]) for r in rows],
        "low": [float(r["bstp_nmix_lwpr"]) for r in rows],
        "close": [float(r["bstp_nmix_prpr"]) for r in rows],
        "volume": [float(r["acml_vol"]) for r in rows],          # 천주
        "value": [float(r["acml_tr_pbmn"]) for r in rows],       # 백만원
    })
    return df.drop_duplicates("date").set_index("date").sort_index()


def main(full: bool = False):
    DATA.mkdir(exist_ok=True)
    c = KISClient()
    today = pd.Timestamp.now().normalize()
    old = None if full or not OUT.exists() else pd.read_csv(OUT, index_col="date", parse_dates=["date"])
    a = START if old is None or old.empty else old.index.max() - pd.Timedelta(days=7)
    new = fetch_range(c, a, today)
    df = new if old is None else pd.concat([old[old.index < a], new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["close"] > 0)]
    df.to_csv(OUT, float_format="%.2f")
    print(f"kospi_daily: {len(df)}행 {df.index.min().date()} ~ {df.index.max().date()} (+{len(new)}행 갱신)")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
