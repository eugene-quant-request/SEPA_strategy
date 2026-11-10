# -*- coding: utf-8 -*-
"""변동성 수축 패턴(VCP)·파워 플레이·단순 신고가 돌파 탐지.

수축 분해(재귀):  베이스 고점 H1 = 창 안 최고가
    L1 = H1 이후 최저가,  H2 = L1 이후 최고가,  L2 = H2 이후 최저가, ...
창 끝에 닿으면 멈춘다. 각 (Hi, Li) 가 한 번의 수축(T), 깊이 = 1 − Li/Hi.
마지막 수축의 고점 Hk 가 피벗. 이 분해는 '저점 이후 최고가'를 쓰므로 3% 미만 잡음 스윙에 흔들리지 않고,
구성상 Hi 는 비증가·Li 는 비감소라 책의 그림(고점은 낮아지고 저점은 높아지는 쐐기)과 같은 모양만 남는다.

베이스 판정은 돌파일 t 의 '전일(t−1)까지' 막대로만 한다 → 돌파일 정보가 패턴 정의에 새지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def contractions(h: np.ndarray, l: np.ndarray, start: int, end: int, swing_pct: float):
    """h,l 배열의 [start, end] 구간에서 H/L 재귀 분해. 반환 [(iH, H, iL, L), ...]."""
    out = []
    pos = start + int(np.nanargmax(h[start:end + 1]))
    while pos < end:
        seg = l[pos + 1:end + 1]
        if seg.size == 0 or np.all(np.isnan(seg)):
            break
        iL = pos + 1 + int(np.nanargmin(seg))
        H, L = h[pos], l[iL]
        if iL >= end:                       # 창 끝이 최저점 → 아직 하락 중, 피벗 없음
            out.append((pos, H, iL, L))
            return out, False
        if 1 - L / H >= swing_pct or not out:
            out.append((pos, H, iL, L))
        seg2 = h[iL + 1:end + 1]
        pos = iL + 1 + int(np.nanargmax(seg2))
    return out, True


def analyze_base(h, l, v, avg50, end: int, cfg: dict):
    """end 인덱스(포함)까지로 VCP 판정. 통과하면 dict, 아니면 None."""
    c = cfg["vcp"]
    start = max(0, end - c["lookback_days"] + 1)
    if end - start < c["min_base_days"] or np.isnan(h[start:end + 1]).all():
        return None
    cs, done = contractions(h, l, start, end, c["swing_pct"])
    if not done or len(cs) < c["min_contractions"] or len(cs) > c["max_contractions"]:
        return None
    base_len = end - cs[0][0]
    if base_len < c["min_base_days"]:
        return None
    depths = [1 - L / H for _, H, _, L in cs]
    if depths[0] > c["first_max"] or depths[-1] > c["last_max"]:
        return None
    if any(depths[i + 1] > depths[i] * c["shrink_tol"] for i in range(len(depths) - 1)):
        return None
    # 피벗은 베이스 고점 근처여야 한다(고점이 계속 낮아지는 하락 쐐기 배제). 육안 검수(CJ 2024-08 등)에서 추가.
    if cs[-1][1] < cs[0][1] * c.get("pivot_near_high", 0.0):
        return None
    iH_last = cs[-1][0]
    vol_last = np.nanmean(v[iH_last:end + 1])
    dry = vol_last / avg50[end] if avg50[end] and not np.isnan(avg50[end]) else np.nan
    if not (dry < c["dryup_ratio"]):
        return None
    return {"pivot": float(cs[-1][1]), "last_low": float(cs[-1][3]), "n_contr": len(cs),
            "depths": depths, "base_len": int(base_len), "dryup": float(dry),
            "base_start_i": int(cs[0][0]), "last_high_i": int(iH_last)}


def power_play(h, l, t: int, cfg: dict):
    """40영업일 내 +100% 급등 → 15~30영업일 깃발(고점 대비 조정 ≤25%) → t 일 깃발 고점 돌파 후보."""
    p = cfg["vcp"]["power_play"]
    lo = t - p["flag_max"]
    if lo - p["run_days"] < 0:
        return None
    win = h[lo:t]
    if np.isnan(win).all():
        return None
    ih = lo + int(np.nanargmax(win))
    flag_len = t - 1 - ih
    if not (p["flag_min"] <= flag_len <= p["flag_max"]):
        return None
    H = h[ih]
    run_low = np.nanmin(l[max(0, ih - p["run_days"]):ih + 1])
    if H / run_low - 1 < p["run_pct"]:
        return None
    flag_low = np.nanmin(l[ih + 1:t])
    if 1 - flag_low / H > p["flag_max_dd"]:
        return None
    return {"pivot": float(H), "last_low": float(flag_low), "n_contr": 1, "depths": [1 - flag_low / H],
            "base_len": int(flag_len), "dryup": np.nan, "base_start_i": int(ih), "last_high_i": int(ih)}


def detect_signals(px: dict, tt_pass: pd.DataFrame, cfg: dict, codes=None, kinds=("vcp", "pp")) -> pd.DataFrame:
    """돌파 신호(date t = 종가 확정일). 체결은 백테스트에서 t+1 시가.
    VCP: t−1 까지의 베이스가 VCP, 종가_t > 피벗 ≥ 종가_{t−1}, 거래량_t ≥ 50일 평균 × breakout_vol, tt_pass_t.
    PP : 파워 플레이 깃발 고점 돌파(같은 거래량 조건)."""
    c = cfg["vcp"]
    H, L, C, V = (px[k] for k in ("adj_high", "adj_low", "adj_close", "adj_volume"))
    avg50 = V.rolling(50, min_periods=40).mean().shift(1)     # 전일까지 50일 평균(돌파일 제외)
    cand = tt_pass & (V >= avg50 * c["breakout_vol"]) & (C > C.shift(1))
    codes = codes if codes is not None else cand.columns[cand.any()]
    idx = C.index
    rows = []
    for code in codes:
        ts = np.flatnonzero(cand[code].values)
        if ts.size == 0:
            continue
        h, l, v, cl, a50 = (x[code].values.astype(float) for x in (H, L, V, C, avg50))
        for t in ts:
            if t < 60:
                continue
            for kind in kinds:
                r = analyze_base(h, l, v, a50, t - 1, cfg) if kind == "vcp" else power_play(h, l, t, cfg)
                if r is None:
                    continue
                if cl[t] > r["pivot"] >= cl[t - 1]:
                    r.update(date=idx[t], code=code, kind=kind, close=cl[t], vol_ratio=v[t] / a50[t],
                             base_start=idx[r.pop("base_start_i")], last_high_date=idx[r.pop("last_high_i")])
                    rows.append(r)
                    break
    cols = ["date", "code", "kind", "pivot", "last_low", "n_contr", "depths", "base_len", "dryup",
            "close", "vol_ratio", "base_start", "last_high_date"]
    return pd.DataFrame(rows, columns=cols).sort_values(["date", "code"]).reset_index(drop=True)


def detect_breakouts(px: dict, tt_pass: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """실험 A/B 용 단순 돌파: tt_pass_t, 종가_t > 직전 N일 최고가, 거래량 ≥ 50일 평균 × vol_mult (첫 돌파일만)."""
    b = cfg["breakout"]
    H, C, V = px["adj_high"], px["adj_close"], px["adj_volume"]
    prior_hi = H.rolling(b["high_lookback"], min_periods=b["high_lookback"]).max().shift(1)
    avg50 = V.rolling(50, min_periods=40).mean().shift(1)
    brk = (C > prior_hi)
    sig = tt_pass & brk & ~brk.shift(1, fill_value=False) & (V >= avg50 * b["vol_mult"])
    st = sig.stack()
    st = st[st]
    df = st.index.to_frame(index=False)
    df.columns = ["date", "code"]
    df["kind"] = "breakout"
    df["pivot"] = [prior_hi.at[d, c] for d, c in zip(df.date, df.code)]
    df["last_low"] = np.nan
    return df


def current_bases(px: dict, tt_pass: pd.DataFrame, cfg: dict, when=None) -> pd.DataFrame:
    """스크리너용: when(기본 최신일)에 tt_pass 인 종목 중 '형성 중인' VCP(돌파 전 포함)."""
    C = px["adj_close"]
    t = len(C.index) - 1 if when is None else C.index.get_loc(when)
    avg50 = px["adj_volume"].rolling(50, min_periods=40).mean()
    rows = []
    for code in tt_pass.columns[tt_pass.iloc[t].fillna(False).values.astype(bool)]:
        h, l, v, a = (x[code].values.astype(float) for x in (px["adj_high"], px["adj_low"], px["adj_volume"], avg50))
        r = analyze_base(h, l, v, a, t, cfg)
        if r is None:
            continue
        r.pop("base_start_i"); r.pop("last_high_i")
        r["code"] = code
        rows.append(r)
    return pd.DataFrame(rows)
