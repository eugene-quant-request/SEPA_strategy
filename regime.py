# -*- coding: utf-8 -*-
"""시장 국면 필터. 모든 값은 t일 종가 기준(신규 진입은 t+1 시가)."""
from __future__ import annotations

import pandas as pd


def ma_regime(level: pd.Series, ma: int, lag: int) -> pd.Series:
    """지수 > 200일선 and 200일선 상승."""
    m = level.rolling(ma, min_periods=ma).mean()
    return (level > m) & (m > m.shift(lag))


def equal_weight_index(close: pd.DataFrame, universe: pd.DataFrame) -> pd.Series:
    """유니버스 동일가중 지수(전일 유니버스 종목의 당일 수익률 평균). 삼전·하이닉스 쏠림 왜곡 확인용."""
    r = close.pct_change(fill_method=None)
    w = universe.shift(1).fillna(False).astype(bool)
    ew = r.where(w).mean(axis=1).fillna(0)
    return (1 + ew).cumprod()


def compute(index_close: pd.DataFrame, close: pd.DataFrame, universe: pd.DataFrame, tt_pass: pd.DataFrame, cfg: dict,
            detector: pd.Series | None = None) -> pd.DataFrame:
    g = cfg["regime"]
    days = close.index
    kospi = index_close["KOSPI"].reindex(days).ffill()
    ew = equal_weight_index(close, universe)
    breadth = tt_pass.sum(axis=1) / universe.sum(axis=1).replace(0, float("nan"))
    out = pd.DataFrame({
        "kospi": kospi,
        "ew_index": ew,
        "breadth": breadth,
        "mkt_ok": ma_regime(kospi, g["ma"], g["rising_lag"]),
        "mkt_ok_ew": ma_regime(ew, g["ma"], g["rising_lag"]),
    }, index=days)
    out["breadth_ok"] = out["breadth"] >= g["breadth_min"]
    if detector is not None:          # 유진 리서치 디텍터(월별 레짐, 월말 확정 → 다음 달부터 사용)
        out["detector_ok"] = detector.reindex(days, method="ffill").shift(1).fillna(False).astype(bool)
    return out
