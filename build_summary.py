# -*- coding: utf-8 -*-
"""결과 요약 페이지 → output/sepa_summary.html

experiments.pkl·이벤트 스터디·주도주·당일 스크린을 모아 한 장짜리 HTML 로 만든다.
숫자는 전부 이 스크립트가 데이터에서 다시 계산한다(conclusions.md 와 같은 원천).
    python build_summary.py
"""
from __future__ import annotations

import glob
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import data_loader as dl
import report as rp
import trend

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
CFG = dl.CFG


def r4(x):
    return None if x is None or pd.isna(x) else round(float(x), 4)


def curve(e: pd.Series, start: str):
    e = e[e.index >= pd.Timestamp(start)].dropna()
    m = e.resample("ME").last()
    return [[d.strftime("%Y-%m"), round(float(v / e.iloc[0]), 4)] for d, v in m.items()]


def perf_rows(R, keys, bench):
    t = rp.perf_table(R, keys, bench)
    return [{"key": k, "label": lab, **{c: r4(v) for c, v in row.items()}}
            for k, (lab, row) in zip([k for k in keys if k in R], t.iterrows())]


def ev(f, mask=None):
    x = f if mask is None else f[mask]
    return {"n": int(len(x)), "x20": r4(x["x20"].mean()), "x60": r4(x["x60"].mean()),
            "x120": r4(x["x120"].mean()), "win60": r4((x["x60"] > 0).mean())}


def main():
    res = pickle.load(open(OUT / "experiments.pkl", "rb"))
    R, bench = res["R"], res["bench"]
    px = dl.load_panel()
    names = dl.load_names(); sector = dl.load_sector()
    uni = dl.universe_mask(px, CFG, names, ["KOSPI"])
    last = px["adj_close"].index[-1]
    D = {"asof": str(last.date()), "start": CFG["data"]["start"]}

    s16 = "2016-06-01"
    D["curves16"] = {
        "B 추세+펀더멘털+돌파 · 손절 8%": curve(R["HB8"]["equity"], s16),
        "A 추세 템플릿만": curve(R["A"]["equity"], s16),
        "D 책 그대로(SEPA 전체)": curve(R["D"]["equity"], s16),
        "KOSPI": curve(bench["KOSPI"], s16),
        "KOSPI 동일가중": curve(bench["KOSPI 동일가중"], s16),
    }
    D["curves11"] = {
        "A 추세 템플릿만": curve(R["A_full"]["equity"], "2011-01-01"),
        "A + 국면 필터": curve(R["A_reg_full"]["equity"], "2011-01-01"),
        "VCP 진입": curve(R["V_full"]["equity"], "2011-01-01"),
        "KOSPI": curve(bench["KOSPI"], "2011-01-01"),
        "KOSPI 동일가중": curve(bench["KOSPI 동일가중"], "2011-01-01"),
    }
    D["exp"] = perf_rows(R, ["A", "B", "HB8", "V", "C", "D", "E", "F", "G", "D_ew", "D_breadth", "D_det", "D_strict", "D_pt"], bench)
    D["exp11"] = perf_rows(R, ["A_full", "A_reg_full", "V_full", "V_reg_full"], bench)
    D["bench16"] = {i: {c: r4(v) for c, v in r.items()} for i, r in rp.bench_table(bench, s16).iterrows()}
    D["bench11"] = {i: {c: r4(v) for c, v in r.items()} for i, r in rp.bench_table(bench, "2011-01-01").iterrows()}
    D["stops"] = perf_rows(R, [f"HB{x}" for x in (5, 7, 8, 10, 12, 15)], bench)
    D["stopsD"] = perf_rows(R, [f"H{x}" for x in (5, 7, 8, 10, 12, 15)], bench)
    y = rp.yearly_table(R, ["A_full", "A_reg_full"], bench)
    D["yearly"] = [{"y": int(i), **{c: r4(v) for c, v in r.items()}} for i, r in y.iterrows()]
    D["d_hold"] = r4(R["D"]["n_hold"].mean())
    tC = R["C"]["trades"]
    D["c_stops"] = {"n": int(len(tC)), "stop": int(tC["reason"].str.startswith("stop").sum()),
                    "within5": int((tC["reason"].str.startswith("stop") & (tC["hold_days"] <= 5)).sum())}

    # 이벤트 스터디
    fa = rp.forward_excess(res["sig_A"], px, uni)
    fv = rp.forward_excess(res["sig_vcp"], px, uni)
    ise = pd.Timestamp(CFG["backtest"]["in_sample_end"])
    fa16, fv16 = fa[fa.date >= s16], fv[fv.date >= s16]
    E = {
        "brk_is": ev(fa, fa.date <= ise), "brk_oos": ev(fa, fa.date > ise),
        "vcp_is": ev(fv, fv.date <= ise), "vcp_oos": ev(fv, fv.date > ise),
        "fund_y": ev(fa16, fa16.fund_pass), "fund_n": ev(fa16, ~fa16.fund_pass),
        "fund_y_is": ev(fa16, fa16.fund_pass & (fa16.date <= ise)), "fund_n_is": ev(fa16, ~fa16.fund_pass & (fa16.date <= ise)),
        "fund_y_oos": ev(fa16, fa16.fund_pass & (fa16.date > ise)), "fund_n_oos": ev(fa16, ~fa16.fund_pass & (fa16.date > ise)),
        "cyc_fund": ev(fa16, fa16.is_cyclical & fa16.fund_pass), "grw_fund": ev(fa16, ~fa16.is_cyclical & fa16.fund_pass),
        "cyc_none": ev(fa16, fa16.is_cyclical & ~fa16.fund_pass), "grw_none": ev(fa16, ~fa16.is_cyclical & ~fa16.fund_pass),
        "cyc_c33": ev(fa16, fa16.is_cyclical & fa16.code33), "grw_c33": ev(fa16, ~fa16.is_cyclical & fa16.code33),
        "vcp_fund": ev(fv16, fv16.fund_pass), "vcp_nofund": ev(fv16, ~fv16.fund_pass),
    }
    q = pd.qcut(fv16["cat_score"].rank(method="first"), 3, labels=["lo", "mid", "hi"])
    for k in ("lo", "mid", "hi"):
        E["cat_" + k] = ev(fv16, q == k)
    D["ev"] = E
    D["n_vcp"] = int(len(res["sig_vcp"])); D["n_brk"] = int(len(res["sig_A"]))

    # 주도주(질문 7)
    tt = trend.compute(px, uni, CFG)["tt_pass"]
    codes, rr = rp.leaders(px, uni, sector, names)
    fd = rp.first_dates(codes, tt, res["sig_vcp"]).set_index("code")
    C = px["adj_close"]
    L = []
    for c in codes:
        s = C[c].loc["2023-01-02":].dropna()
        half = s[s >= s.iloc[0] * (1 + (s.max() / s.iloc[0] - 1) / 2)].index.min()
        caught = []
        for lab, key in (("A", "A"), ("B", "B")):
            t = R[key]["trades"]
            t = t[(t.code == c) & (t.entry_date >= "2023-01-01")]
            for _, x in t.iterrows():
                caught.append({"exp": lab, "in": str(x.entry_date.date()),
                               "out": (str(x.exit_date.date()) if pd.notna(x.exit_date) else "보유중"),
                               "ret": r4(x.ret), "why": x.reason})
        L.append({"code": c, "name": names.get(c, c), "ret": r4(rr[c]),
                  "tt": str(fd.at[c, "첫 추세템플릿 통과"].date()) if pd.notna(fd.at[c, "첫 추세템플릿 통과"]) else "",
                  "half": str(half.date()), "peak": str(s.idxmax().date()),
                  "vcp": int(fd.at[c, "VCP 신호 수"]), "trades": caught})
    D["leaders"] = L

    # 당일 스크린
    sf = sorted(glob.glob(str(OUT / "screen_*.csv")))[-1]
    sc = pd.read_csv(sf, dtype={"코드": str})
    keep = ["코드", "종목명", "업종", "경기민감", "RS", "최신분기", "EPS_YoY_q0", "매출_YoY_q0", "fund_pass", "패턴", "VCP 수축",
            "피벗대비", "신고가60일", "수급20일/시총", "테마"]
    sc = sc[keep].replace({np.nan: None})
    D["screen"] = {"date": Path(sf).stem[-8:], "rows": sc.to_dict("records")}

    html = (ROOT / "summary_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(D, ensure_ascii=False, default=str))
    (OUT / "sepa_summary.html").write_text(html, encoding="utf-8")
    print("→", OUT / "sepa_summary.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
