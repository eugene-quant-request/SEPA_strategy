# -*- coding: utf-8 -*-
"""실험 A~H 실행 → output/experiments.pkl, trades.csv, 리포트(report.py).

    python run_backtest.py            # 신호 캐시가 있으면 재사용
    python run_backtest.py --rebuild  # 신호 다시 계산
"""
from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import catalyst
import data_loader as dl
import fundamentals as fm
import regime as rg
import trend
import vcp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
CACHE = dl.DATA / "cache"; CACHE.mkdir(exist_ok=True)
CFG = dl.CFG
FUND_START = "2016-06-01"      # DART 분기재무 YoY 가 2016Q1(5월 공시)부터 → 펀더멘털 실험 공통 시작일


def log(*a):
    dl.log(*a)


# ───────────────────────── 준비 ─────────────────────────
def prepare(markets, rebuild=False):
    key = "_".join(markets)
    f = CACHE / f"ctx_{key}.pkl"
    if f.exists() and not rebuild:
        return pickle.load(open(f, "rb"))
    px = dl.load_panel()
    names = dl.load_names()
    uni = dl.universe_mask(px, CFG, names, markets)
    tr = trend.compute(px, uni, CFG)
    log("trend", key, "tt_pass stock-days", int(tr["tt_pass"].values.sum()))
    brk = vcp.detect_breakouts(px, tr["tt_pass"], CFG)
    log("breakouts", len(brk))
    vs = vcp.detect_signals(px, tr["tt_pass"], CFG)
    log("vcp/pp signals", len(vs), vs["kind"].value_counts().to_dict() if len(vs) else {})
    ctx = {"markets": markets, "uni": uni, "tt_pass": tr["tt_pass"], "rs": tr["rs"], "stage": tr["stage"],
           "ma50": tr["ma50"], "ma200": tr["ma200"], "brk": brk, "vcp": vs}
    pickle.dump(ctx, open(f, "wb"))
    return ctx


def fund_table():
    """포인트인타임 재무 + 흑자전환을 성장으로 보지 않는 엄격판 fund_pass(fund_pass_strict)."""
    fq = pd.read_parquet(dl.DATA / "fin_quarterly.parquet")
    days = dl.trading_days()
    pit = fm.point_in_time(fq, days, CFG)
    c2 = copy.deepcopy(CFG); c2["fundamentals"]["turnaround_as_growth"] = False
    strict = fm.point_in_time(fq, days, c2)[["code", "eff", "fund_pass"]].rename(columns={"fund_pass": "fund_pass_strict"})
    return pit.merge(strict, on=["code", "eff"], how="left")


def attach(sig: pd.DataFrame, ctx, pit, sector, names, catalyst_parts=None) -> pd.DataFrame:
    """신호 행에 RS·단계·펀더멘털(그 시점 최신 분기)·업종·촉매 점수를 붙인다."""
    s = sig.copy()
    s["rs"] = [ctx["rs"].at[d, c] for d, c in zip(s.date, s.code)]
    s["stage"] = [ctx["stage"].at[d, c] for d, c in zip(s.date, s.code)]
    p = pit.rename(columns={"eff": "date"}).sort_values("date")
    s = pd.merge_asof(s.sort_values("date"), p, on="date", by="code", direction="backward",
                      tolerance=pd.Timedelta(days=200))
    s["fund_pass"] = s["fund_pass"].astype("boolean").fillna(False).astype(bool)
    s["fund_pass_strict"] = s["fund_pass_strict"].astype("boolean").fillna(False).astype(bool)
    s["code33"] = s["code33"].astype("boolean").fillna(False).astype(bool)
    s["sector"] = s["code"].map(sector)
    s["is_cyclical"] = s["code"].map(dl.cyclical_map()).fillna(False).astype(bool)
    s["name"] = s["code"].map(names)
    if catalyst_parts is not None:
        sc, parts = catalyst.score(s, catalyst_parts, CFG)
        s["cat_score"] = sc
        s[["cat_new_high", "cat_flow", "cat_theme"]] = parts.values
    else:
        s["cat_score"] = np.nan
    s["priority"] = s["rs"].fillna(0) + 0.001 * s["cat_score"].fillna(0)
    return s.sort_values(["date", "code"]).reset_index(drop=True)


# ───────────────────────── 실험 ─────────────────────────
def experiments(rebuild=False):
    px = dl.load_panel()
    names = dl.load_names()
    sector = dl.load_sector()
    pit = fund_table()
    idx = pd.read_parquet(dl.DATA / "index_close.parquet")
    ctx = prepare(["KOSPI"], rebuild)
    ctx2 = prepare(["KOSPI", "KOSDAQ"], rebuild)

    det = None
    if CFG["paths"].get("regime_csv"):
        det = detector_ok(ROOT / CFG["paths"]["regime_csv"], idx["KOSPI"])
    reg = rg.compute(idx, px["adj_close"], ctx["uni"], ctx["tt_pass"], CFG, det)
    reg2 = rg.compute(idx, px["adj_close"], ctx2["uni"], ctx2["tt_pass"], CFG, det)

    cat_parts = catalyst_parts(px, ctx["uni"], pd.concat([ctx["vcp"], ctx2["vcp"]]))

    A_sig = attach(ctx["brk"], ctx, pit, sector, names)
    V_sig = attach(ctx["vcp"], ctx, pit, sector, names, cat_parts)
    V2_sig = attach(ctx2["vcp"], ctx2, pit, sector, names, cat_parts)
    A_sig.to_parquet(CACHE / "sig_A.parquet"); V_sig.to_parquet(CACHE / "sig_vcp.parquet")

    ok = lambda s, r, col="mkt_ok": s[s["date"].map(r[col]).fillna(False).astype(bool)]
    run = lambda sig, label, c=ctx, start=None, **k: bt.run(sig, px, c["ma50"], c["ma200"], CFG, start=start, label=label, **k)

    E_sig = V_sig.copy(); E_sig["priority"] = E_sig["cat_score"].fillna(0) + 0.001 * E_sig["rs"].fillna(0)
    Dsig = ok(V_sig[V_sig.fund_pass], reg)
    R = {}
    # 전체 기간(2011~): 펀더멘털 없는 전략
    R["A_full"] = run(A_sig, "A 추세템플릿+돌파 (2011~)")
    R["A_reg_full"] = run(ok(A_sig, reg), "A + 국면필터 (2011~)")
    R["V_full"] = run(V_sig, "VCP 진입(펀더멘털 없음) (2011~)")
    R["V_reg_full"] = run(ok(V_sig, reg), "VCP + 국면필터 (2011~)")
    # 공통 기간(2016-06~)
    s = FUND_START
    R["A"] = run(A_sig, "A 추세템플릿+돌파", start=s)
    R["B"] = run(A_sig[A_sig.fund_pass], "B A+펀더멘털", start=s)
    R["V"] = run(V_sig, "C0 VCP만(펀더멘털 없음)", start=s)
    R["C"] = run(V_sig[V_sig.fund_pass], "C B+VCP 피벗", start=s)
    R["D"] = run(Dsig, "D C+국면필터", start=s)
    R["E"] = run(ok(E_sig[E_sig.fund_pass], reg), "E D+촉매 우선순위", start=s)
    R["F"] = run(ok(V2_sig[V2_sig.fund_pass], reg2), "F D+코스닥", c=ctx2, start=s)
    R["G"] = run(Dsig[~Dsig.is_cyclical], "G D−경기민감주", start=s)
    for sp in (0.05, 0.07, 0.08, 0.10, 0.12, 0.15):
        c_ = copy.deepcopy(CFG); c_["backtest"]["stop_max"] = max(sp, CFG["backtest"]["stop_max"])
        R[f"H{int(sp*100)}"] = bt.run(Dsig, px, ctx["ma50"], ctx["ma200"], c_, start=s, stop_pct=sp, label=f"H 손절 {int(sp*100)}% (D)")
        R[f"HB{int(sp*100)}"] = bt.run(A_sig[A_sig.fund_pass], px, ctx["ma50"], ctx["ma200"], c_, start=s, stop_pct=sp,
                                       label=f"H 손절 {int(sp*100)}% (B, 표본 큼)")
    # 추가 변형
    R["D_ew"] = run(ok(V_sig[V_sig.fund_pass], reg, "mkt_ok_ew"), "D' 동일가중 국면", start=s)
    R["D_breadth"] = run(ok(V_sig[V_sig.fund_pass], reg, "breadth_ok"), "D'' 브레드스 국면", start=s)
    if "detector_ok" in reg:
        R["D_det"] = run(ok(V_sig[V_sig.fund_pass], reg, "detector_ok"), "D''' 유진 리서치 디텍터 국면", start=s)
    R["D_strict"] = run(ok(V_sig[V_sig.fund_pass_strict], reg), "D (흑자전환 불인정)", start=s)
    cfg_pt = copy.deepcopy(CFG); cfg_pt["backtest"]["partial_take"]["enabled"] = True
    R["D_pt"] = bt.run(Dsig, px, ctx["ma50"], ctx["ma200"], cfg_pt, start=s, label="D + 강세 속 분할매도")

    bench = pd.DataFrame({"KOSPI": idx["KOSPI"], "KOSPI200": idx["KOSPI200"], "KOSPI 동일가중": reg["ew_index"]}).reindex(px["adj_close"].index).ffill()
    res = {"R": R, "bench": bench, "reg": reg, "sig_A": A_sig, "sig_vcp": V_sig, "sig_vcp2": V2_sig}
    pickle.dump(res, open(OUT / "experiments.pkl", "wb"))
    return res


def catalyst_parts(px, uni, vsig):
    parts = {"new_high": catalyst.new_high_flag(px["adj_close"], CFG["catalyst"]["new_high_days"])}
    codes = sorted(set(vsig["code"]))
    flows = dl.load_flows(codes)
    if flows.shape[1]:
        parts["flow"] = catalyst.flow_ratio(flows, px["mktcap"], CFG["catalyst"]["flow_days"])
        log("flows loaded", flows.shape[1], "/", len(codes))
    th = dl.load_themes()
    if len(th):
        parts["theme"] = catalyst.theme_rel(px["adj_close"], uni, th, CFG["catalyst"]["theme_rel_days"])
    return parts


def detector_ok(path, kospi):
    """유진 리서치 디텍터 월별 레짐 → '우호 레짐' 여부. 우호 판정은 표본 내(2004~2019) 다음 달 KOSPI 수익률 평균 > 0 인 레짐.
    주의: 디텍터 모델 자체는 전체 표본으로 적합돼 있어 이 변형은 참고용이다."""
    r = pd.read_csv(path, index_col=0, parse_dates=True, encoding="utf-8-sig")
    k = kospi.resample("ME").last()
    fwd = k.pct_change().shift(-1).reindex(r.index, method="ffill")
    ins = r.index <= "2019-12-31"
    good = fwd[ins].groupby(r.loc[ins, "regime"]).mean()
    log("detector regimes fwd 1m (in-sample)", good.round(4).to_dict())
    return r["regime"].map(lambda x: bool(good.get(x, 0) > 0))


if __name__ == "__main__":
    res = experiments("--rebuild" in sys.argv)
    rows = {k: {"label": v["label"], **bt.perf(v["equity"], v["trades"], CFG["backtest"]["short_ban"])} for k, v in res["R"].items()}
    print(pd.DataFrame(rows).T.to_string())
