# -*- coding: utf-8 -*-
"""추세 템플릿(8조건)·RS 순위·4단계 판정.

입력은 수정주가 wide 패널(행=일자, 열=종목). 모든 계산은 t일 종가까지의 정보만 쓴다(체결은 t+1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rs_raw(close: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """IBD 근사: 0.4·R63 + 0.2·R126 + 0.2·R189 + 0.2·R252 (최근 분기 가중 2배)."""
    out = 0
    for n, w in weights.items():
        out = out + w * (close / close.shift(int(n)) - 1)
    return out


def rs_rank(raw: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """유니버스(eligible) 안의 백분위 0~99. 지수 대비가 아니다(삼전·하이닉스 쏠림 회피)."""
    r = raw.where(eligible)
    return (r.rank(axis=1, pct=True) * 99).round()


def compute(px: dict, eligible: pd.DataFrame, cfg: dict) -> dict:
    """px: load_panel() 결과. eligible: 유니버스 마스크(유동성·특이종목 제외).
    반환: c1..c8, tt_pass, rs, stage, ma50/150/200, hi52/lo52, n_hist."""
    t = cfg["trend"]
    c = px["adj_close"]
    ma50 = c.rolling(t["ma_short"], min_periods=t["ma_short"]).mean()
    ma150 = c.rolling(t["ma_mid"], min_periods=t["ma_mid"]).mean()
    ma200 = c.rolling(t["ma_long"], min_periods=t["ma_long"]).mean()
    hi52 = px["adj_high"].rolling(252, min_periods=200).max()
    lo52 = px["adj_low"].rolling(252, min_periods=200).min()
    n_hist = c.notna().cumsum()
    has_hist = n_hist >= t["min_history"]

    rs = rs_rank(rs_raw(c, t["rs_weights"]), eligible & has_hist)

    lag = t["ma200_rising_lag"]
    cond = {
        "c1": (c > ma150) & (c > ma200),
        "c2": ma150 > ma200,
        "c3": ma200 > ma200.shift(lag),
        "c4": (ma50 > ma150) & (ma50 > ma200),
        "c5": c > ma50,
        "c6": c >= lo52 * t["above_52w_low"],
        "c7": c >= hi52 * t["near_52w_high"],
        "c8": rs >= t["rs_min"],
    }
    tt = has_hist & eligible
    for v in cond.values():
        tt = tt & v

    # ── 단계 판정 규칙 ──
    # 2단계: 종가 > 200일선, 200일선 상승(lag), 50 > 150 > 200 정배열
    # 4단계: 종가 < 200일선, 200일선 하락
    # 3단계: 2·4단계가 아니고 200일선이 아직 하락 전환 전(기울기 ≥ 0)이며 최근 126일 안에 2단계였던 종목(천정/분산)
    # 1단계: 그 밖(하락 후 바닥 다지기, 200일선 횡보)
    slope = ma200 / ma200.shift(lag) - 1
    s2 = (c > ma200) & (slope > 0) & (ma50 > ma150) & (ma150 > ma200)
    s4 = (c < ma200) & (slope < 0)
    was2 = s2.astype(float).rolling(126, min_periods=1).max() > 0
    s3 = ~s2 & ~s4 & (slope >= 0) & was2
    stage = pd.DataFrame(1.0, index=c.index, columns=c.columns)
    stage = stage.mask(s3, 3).mask(s4, 4).mask(s2, 2).where(ma200.notna())

    out = dict(cond)
    out.update(tt_pass=tt, rs=rs, stage=stage, ma50=ma50, ma150=ma150, ma200=ma200,
               hi52=hi52, lo52=lo52, n_hist=n_hist, ipo_flag=~has_hist & c.notna())
    return out
