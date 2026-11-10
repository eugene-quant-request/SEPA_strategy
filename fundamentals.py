# -*- coding: utf-8 -*-
"""포인트인타임 분기 성장·가속·코드 33.

입력 data/fin_quarterly.parquet (data_loader.build_fundamentals): code, fy, q, sales, op, ni(단일분기 십억원), avail(접수일).
각 일자에 '그 시점에 공시로 알려진 최신 분기'를 붙인다. 접수일 다음 영업일부터 사용(룩어헤드 금지).
성장률은 YoY 만 쓴다(4분기 일회성 비용 때문에 QoQ 금지).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _yoy(cur, prev, min_base):
    g = (cur - prev) / prev.abs()
    return g.where(prev >= min_base)          # 전년 동기 적자·미미 → NaN


def quarterly_metrics(fq: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = cfg["fundamentals"]
    fq = fq.sort_values(["code", "fy", "q"]).copy()
    prev = fq.set_index(["code", "fy", "q"])[["sales", "op", "ni"]]
    prev.index = pd.MultiIndex.from_tuples([(c, fy + 1, q) for c, fy, q in prev.index], names=prev.index.names)
    p = prev.reindex(pd.MultiIndex.from_frame(fq[["code", "fy", "q"]]))
    fq["sales_prev"], fq["op_prev"], fq["ni_prev"] = p["sales"].values, p["op"].values, p["ni"].values
    fq["eps_yoy"] = _yoy(fq["ni"], fq["ni_prev"], f["min_base_bil"])
    fq["op_yoy"] = _yoy(fq["op"], fq["op_prev"], f["min_base_bil"])
    fq["sales_yoy"] = _yoy(fq["sales"], fq["sales_prev"], f["min_base_bil"])
    fq["turnaround"] = (fq["ni_prev"] <= 0) & (fq["ni"] > 0)
    fq["opm"] = (fq["op"] / fq["sales"]).where(fq["sales"] > 0)
    fq["opm_yoy_change"] = fq["opm"] - (fq["op_prev"] / fq["sales_prev"]).where(fq["sales_prev"] > 0)

    # 직전 분기 이력(q1..q3). 분기가 연속일 때만 이어 붙인다.
    fq["qn"] = fq["fy"] * 4 + fq["q"]
    g = fq.groupby("code")
    for k in (1, 2, 3):
        cont = (g["qn"].shift(k) == fq["qn"] - k)
        for col in ("eps_yoy", "sales_yoy", "opm"):
            fq[f"{col}_q{k}"] = g[col].shift(k).where(cont)
    for col in ("eps_yoy", "sales_yoy", "opm"):
        fq[f"{col}_q0"] = fq[col]

    # 판정용 EPS 성장률: 적자→흑자 전환(turnaround)은 +∞ 로 본다(표시값은 NaN 유지). config 로 끌 수 있다.
    if f.get("turnaround_as_growth", True):
        ta = fq["turnaround"]
        tflag = {0: ta}
        for k in (1, 2, 3):
            tflag[k] = g["turnaround"].shift(k).where(g["qn"].shift(k) == fq["qn"] - k).astype("boolean").fillna(False).astype(bool)
        for k in (0, 1, 2, 3):
            fq[f"_eg_q{k}"] = fq[f"eps_yoy_q{k}"].mask(tflag[k], np.inf)
    else:
        for k in (0, 1, 2, 3):
            fq[f"_eg_q{k}"] = fq[f"eps_yoy_q{k}"]
    acc = lambda col: (fq[f"{col}_q0"] > fq[f"{col}_q1"]) & (fq[f"{col}_q1"] > fq[f"{col}_q2"]) & (fq[f"{col}_q2"] > fq[f"{col}_q3"])
    fq["eps_accel"] = (fq["_eg_q0"] > fq["_eg_q1"]) | (fq["_eg_q0"] == np.inf)   # 흑자 전환 분기는 가속으로 본다
    fq["code33"] = acc("_eg") & acc("sales_yoy") & acc("opm")
    base = (fq["_eg_q0"] >= f["eps_yoy_min"]) & (fq["sales_yoy_q0"] >= f["sales_yoy_min"])
    fq["fund_pass"] = base & (fq["eps_accel"] | fq["code33"]) if f["require_accel_or_code33"] else base
    fq["decel_warning"] = ((fq["eps_yoy_q0"] < fq["eps_yoy_q1"]) & (fq["eps_yoy_q1"] < fq["eps_yoy_q2"])
                           & (fq["eps_yoy_q0"] <= fq["eps_yoy_q1"] * f["decel_ratio"]) & (fq["eps_yoy_q1"] > 0))
    fq["q4_flag"] = fq["q"] == 4           # 4분기 판정은 경고용(단독 매도 사유 아님)
    return fq


FIELDS = ["fy", "q", "eps_yoy_q0", "eps_yoy_q1", "eps_yoy_q2", "sales_yoy_q0", "sales_yoy_q1", "sales_yoy_q2",
          "opm_q0", "opm_q1", "opm_q2", "opm_yoy_change", "eps_accel", "code33", "fund_pass", "decel_warning",
          "turnaround", "q4_flag", "avail"]


def point_in_time(fq: pd.DataFrame, days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """분기 레코드 → 사용 가능일(접수일 다음 영업일) 기준 레코드 테이블.
    같은 종목에서 나중에 공시됐지만 더 오래된 분기(지연·정정 공시)는 버린다 → 최신 분기만 ffill."""
    m = quarterly_metrics(fq, cfg)
    m = m[m["avail"].notna()].copy()
    lag = cfg["fundamentals"]["lag_days"]
    pos = days.searchsorted(m["avail"].values, side="right") + (lag - 1)
    m = m[pos < len(days)].copy()
    m["eff"] = days[pos[pos < len(days)]]
    m = m.sort_values(["code", "eff", "qn"])
    m["qmax"] = m.groupby("code")["qn"].cummax()
    m = m[m["qn"] >= m["qmax"]]
    m = m.drop_duplicates(["code", "eff"], keep="last")
    return m[["code", "eff", "qn"] + FIELDS]


def daily(pit: pd.DataFrame, field: str, days: pd.DatetimeIndex, codes, max_age_days: int = 200) -> pd.DataFrame:
    """레코드 테이블 → 일자×종목 wide (ffill). 최신 공시가 max_age_days(달력일) 보다 오래되면 NaN(공시 중단 종목)."""
    w = pit.pivot(index="eff", columns="code", values=field).reindex(columns=codes)
    w = w.reindex(days.union(w.index)).ffill().reindex(days)
    eff = pit.pivot(index="eff", columns="code", values="eff").reindex(columns=codes)
    eff = eff.reindex(days.union(eff.index)).ffill().reindex(days)
    age = (pd.Series(days, index=days).values[:, None] - eff.values).astype("timedelta64[D]").astype(float)
    return w.where(age <= max_age_days)
