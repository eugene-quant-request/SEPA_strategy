# -*- coding: utf-8 -*-
"""일일 스크리너 → output/screen_YYYYMMDD.csv

    python run_daily.py            # 캐시된 패널로 최신일 스크린
    python run_daily.py --update   # 한투 API 로 지수·종목 일봉을 최신일까지 이어 받고 패널 재구성 후 스크린
    python run_daily.py --kosdaq   # 코스닥 포함
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import catalyst
import data_loader as dl
import fundamentals as fm
import trend
import vcp

ROOT = Path(__file__).resolve().parent
CFG = dl.CFG


def main():
    if "--update" in sys.argv:
        # 가격 소스는 한투 API (KRX Data Marketplace 는 자동화 수집 제한)
        dl.fetch_index_kis(); dl.fetch_kis_prices(list(dl.kis_stocks().query("delist_dt != delist_dt").index)); dl.build_panel_kis()
    markets = ["KOSPI", "KOSDAQ"] if "--kosdaq" in sys.argv else ["KOSPI"]
    px = dl.load_panel()
    names = dl.load_names()
    sector = dl.load_sector()
    cycmap = dl.cyclical_map()
    uni = dl.universe_mask(px, CFG, names, markets)
    tr = trend.compute(px, uni, CFG)
    days = px["adj_close"].index
    t = days[-1]
    tt = tr["tt_pass"].loc[t]
    codes = tt.index[tt.fillna(False).astype(bool)]

    # 형성 중인 VCP(돌파 전 포함) + 오늘 돌파 신호
    bases = vcp.current_bases(px, tr["tt_pass"], CFG).set_index("code") if len(codes) else pd.DataFrame()
    brk_today = vcp.detect_signals(px, tr["tt_pass"].iloc[-400:], CFG, codes=codes) if len(codes) else pd.DataFrame()
    brk_today = brk_today[brk_today["date"] == t].set_index("code") if len(brk_today) else pd.DataFrame()

    pit = fm.point_in_time(pd.read_parquet(dl.DATA / "fin_quarterly.parquet"), days, CFG)
    last = pit[pit["eff"] <= t].sort_values("eff").groupby("code").tail(1).set_index("code")
    last = last[(t - last["eff"]).dt.days <= 200]

    th = dl.load_themes()
    core = th[(th["exposure_level"] == "Core") & (th["sub_theme_name_kr"] != "미분류")].assign(code=lambda d: d["stock_code"].str.zfill(6)) if len(th) else pd.DataFrame(columns=["code", "sub_theme_name_kr"])
    themes = core.groupby("code")["sub_theme_name_kr"].apply(lambda s: ", ".join(sorted(set(s))))
    nh = catalyst.new_high_flag(px["adj_close"], CFG["catalyst"]["new_high_days"]).loc[t]
    thr = catalyst.theme_rel(px["adj_close"], uni, th, CFG["catalyst"]["theme_rel_days"]).loc[t] if len(th) else pd.Series(dtype=float)
    try:
        fl = dl.kis_recent_flows(list(codes))
        f20 = fl[fl.index <= t].tail(CFG["catalyst"]["flow_days"]).sum()
        flr = f20 / px["mktcap"].loc[t].reindex(f20.index)
    except Exception as e:                      # 한투 호출 실패 시 수급 칸만 비운다
        print("수급 조회 실패:", e); flr = pd.Series(dtype=float)

    close = px["adj_close"].loc[t]
    rows = []
    for c in codes:
        f = last.loc[c] if c in last.index else None
        b = bases.loc[c] if len(bases) and c in bases.index else None
        k = brk_today.loc[c] if len(brk_today) and c in brk_today.index else None
        src = k if k is not None else b
        pivot = src["pivot"] if src is not None else np.nan
        low = src["last_low"] if src is not None else np.nan
        stop = max(close[c] * (1 - CFG["backtest"]["stop_pct"]), low if pd.notna(low) and low < close[c] else 0)
        stop = max(stop, close[c] * (1 - CFG["backtest"]["stop_max"]))
        cyc = bool(cycmap.get(c, False))
        g = lambda k_: (f[k_] if f is not None else np.nan)
        rows.append({
            "코드": c, "종목명": names.get(c, ""), "업종": sector.get(c), "경기민감": cyc,
            **{k_: bool(tr[k_].at[t, c]) for k_ in ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")},
            "RS": tr["rs"].at[t, c], "stage": tr["stage"].at[t, c], "종가": close[c],
            "최신분기": (f"{int(f['fy'])}Q{int(f['q'])}" if f is not None else ""),
            "EPS_YoY_q0": g("eps_yoy_q0"), "EPS_YoY_q1": g("eps_yoy_q1"), "EPS_YoY_q2": g("eps_yoy_q2"),
            "매출_YoY_q0": g("sales_yoy_q0"), "매출_YoY_q1": g("sales_yoy_q1"), "매출_YoY_q2": g("sales_yoy_q2"),
            "code33": bool(g("code33")) if f is not None else False,
            "fund_pass": bool(g("fund_pass")) if f is not None else False,
            "감속경고": bool(g("decel_warning")) if f is not None else False,
            "패턴": (src["kind"] if k is not None else ("vcp" if b is not None else "")),
            "오늘돌파": k is not None,
            "VCP 수축": ("→".join(f"{x*100:.0f}%" for x in src["depths"]) if src is not None else ""),
            "피벗": pivot, "피벗대비": (close[c] / pivot - 1) if pd.notna(pivot) else np.nan,
            "제안손절가": stop, "손절폭": 1 - stop / close[c],
            "신고가60일": bool(nh.get(c, False)), "수급20일/시총": flr.get(c, np.nan), "테마상대20일": thr.get(c, np.nan),
            "테마": themes.get(c, ""),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["오늘돌파", "fund_pass", "RS"], ascending=False)
    out = ROOT / "output" / f"screen_{t.strftime('%Y%m%d')}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{t.date()} 추세템플릿 통과 {len(df)}  VCP형성 {int((df['패턴']!='').sum()) if len(df) else 0}  "
          f"오늘돌파 {int(df['오늘돌파'].sum()) if len(df) else 0}  펀더멘털통과 {int(df['fund_pass'].sum()) if len(df) else 0} → {out.name}")


if __name__ == "__main__":
    main()
