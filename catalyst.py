# -*- coding: utf-8 -*-
"""촉매·수급 보조 점수(필터 아님, 동시 신호 우선순위와 상·하위 성과 비교에만 쓴다).

구성
- new_high : 최근 60일 안에 52주 신고가(종가 기준) 갱신 여부
- flow     : (외국인+기관 20일 순매수대금) / 시총  — 신호 종목만 수급을 받아 온다
- theme    : Theme Master Core 소속 테마의 20일 동일가중 수익률 − 유니버스 동일가중 20일 수익률 (소속 테마 중 최대)
비활성
- 잠정실적 서프라이즈·추정치 상향 : 과거 컨센서스 데이터가 없어 계산하지 않는다(추정값으로 채우지 않음).
주의
- Theme Master 는 2026-05 스냅샷이다. 과거 시점 테마 소속을 알 수 없어 테마 점수에는 룩어헤드가 있다 → 리포트에 명시.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def new_high_flag(close: pd.DataFrame, days: int) -> pd.DataFrame:
    hi = close.rolling(252, min_periods=200).max()
    at_high = close >= hi
    return at_high.astype(float).rolling(days, min_periods=1).max() > 0


def flow_ratio(flows: pd.DataFrame, mktcap: pd.DataFrame, days: int) -> pd.DataFrame:
    f = flows.reindex(mktcap.index).fillna(0).rolling(days, min_periods=days).sum()
    return f / mktcap.reindex(columns=f.columns)


def theme_rel(close: pd.DataFrame, universe: pd.DataFrame, themes: pd.DataFrame, days: int) -> pd.DataFrame:
    core = themes[(themes["exposure_level"] == "Core") & (themes["sub_theme_name_kr"] != "미분류")][["sub_theme_name_kr", "stock_code"]].dropna()
    core["stock_code"] = core["stock_code"].str.zfill(6)
    r = close / close.shift(days) - 1
    base = r.where(universe).mean(axis=1)
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for th, g in core.groupby("sub_theme_name_kr"):
        mem = [c for c in g["stock_code"] if c in close.columns]
        if len(mem) < 3:
            continue
        rel = r[mem].mean(axis=1) - base
        for c in mem:
            out[c] = np.fmax(out[c].values, rel.values)
    return out


def score(sig: pd.DataFrame, parts: dict, cfg: dict) -> pd.Series:
    """신호 행마다 점수. 연속형은 같은 날 후보 안 백분위(없으면 0.5 중립)."""
    w = cfg["catalyst"]["weights"]
    s = sig[["date", "code"]].copy()
    for k in ("new_high", "flow", "theme"):
        p = parts.get(k)
        s[k] = [p.at[d, c] if (p is not None and c in p.columns and d in p.index) else np.nan
                for d, c in zip(s.date, s.code)]
    s["new_high"] = s["new_high"].astype(float)
    for k in ("flow", "theme"):
        s[k + "_pct"] = s.groupby("date")[k].rank(pct=True).fillna(0.5)
    total = w["flow"] * s["flow_pct"] + w["new_high"] * s["new_high"].fillna(0) + w["theme"] * s["theme_pct"]
    return total.values, s[["new_high", "flow", "theme"]]
