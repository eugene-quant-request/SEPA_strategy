# -*- coding: utf-8 -*-
"""이벤트 기반 일별 포트폴리오 시뮬레이션.

타이밍(하루 d 의 순서)
 1) 시가: 전일 종가로 확정된 청산 주문(추세 이탈·분할매도 없음) 체결 → 전일 신호 신규 진입(시가+슬리피지)
 2) 장중: 손절가 체크. 시가가 이미 손절가 아래(갭 하락)면 손절가가 아니라 '시가'로 체결.
 3) 종가: 평가, 손절선 상향(2R→본전, 3R→+1R), 추세 이탈 판정(다음날 시가 매도), 상장폐지 마지막 날은 종가 청산.

가격은 수정주가(배당 미포함 가격지수)라 벤치마크(KOSPI 가격지수)와 같은 기준이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sell_tax_table(cfg):
    t = pd.DataFrame(cfg["backtest"]["sell_tax"], columns=["from", "KOSPI", "KOSDAQ"])
    t["from"] = pd.to_datetime(t["from"])
    return t.sort_values("from")


def _tax(tbl, d, is_kospi):
    row = tbl[tbl["from"] <= d].iloc[-1]
    return row["KOSPI"] if is_kospi else row["KOSDAQ"]


def run(signals: pd.DataFrame, px: dict, ma50: pd.DataFrame, ma200: pd.DataFrame, cfg: dict,
        start=None, end=None, stop_pct=None, label="") -> dict:
    """signals: date(신호일=종가 확정일), code, priority(클수록 우선), last_low(수축 저점, 없으면 NaN), 그 외 메타.
    반환: equity(Series), trades(DataFrame), daily(DataFrame: 보유 수·현금비중)."""
    b = cfg["backtest"]
    stop_pct = stop_pct if stop_pct is not None else b["stop_pct"]
    O, H, L, C, V = (px[k] for k in ("adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"))
    days = C.index
    start = pd.Timestamp(start or b["start"]); end = pd.Timestamp(end) if end else days[-1]
    d0 = days.searchsorted(start); d1 = days.searchsorted(end, side="right")
    col = {c: i for i, c in enumerate(C.columns)}
    o, h, l, c, v = (x.values for x in (O, H, L, C, V))
    c_ff = C.ffill().values
    m50, m200 = ma50.values, ma200.values
    avg50 = V.rolling(50, min_periods=40).mean().values
    kospi = px["is_kospi"].reindex_like(C).ffill().values
    last_valid = C.apply(lambda s: s.last_valid_index())
    last_i = {cc: days.get_loc(d) for cc, d in last_valid.items() if d is not None}
    taxes = sell_tax_table(cfg)
    slip, comm = b["slippage"], b["commission"]
    pt = b["partial_take"]

    sig = signals[(signals["date"] >= days[max(d0 - 1, 0)]) & (signals["date"] < days[min(d1, len(days) - 1)])]
    sig_by_day = {d: g.sort_values("priority", ascending=False) for d, g in sig.groupby("date")}

    cash = float(b["initial_capital"])
    pos = {}          # code → dict
    pending_exit = {}  # code → reason
    trades, eq, nh = [], [], []

    def close_pos(code, i, px_exec, reason, frac=1.0):
        nonlocal cash
        p = pos[code]
        tax = _tax(taxes, days[i], kospi[i, col[code]] == 1)
        sh = p["shares"] * frac
        proceeds = sh * px_exec * (1 - slip - comm - tax)
        cash += proceeds
        cost = sh * p["entry_cost_ps"]
        trades.append({**p["meta"], "code": code, "entry_date": days[p["i_entry"]], "entry_px": p["entry"],
                       "stop0": p["stop0"], "exit_date": days[i], "exit_px": px_exec, "reason": reason,
                       "frac": frac, "ret": proceeds / cost - 1, "r_mult": (px_exec - p["entry"]) / p["R"],
                       "hold_days": i - p["i_entry"], "weight0": p["weight0"], "max_gain": p["max_c"] / p["entry"] - 1})
        if frac >= 1.0:
            del pos[code]
        else:
            p["shares"] -= sh; p["partial_done"] = True

    prev_equity = cash
    for i in range(d0, d1):
        d = days[i]
        # 1) 시가 청산
        for code, reason in list(pending_exit.items()):
            if code in pos:
                j = col[code]
                if not np.isnan(o[i, j]):
                    close_pos(code, i, o[i, j], reason)
                    del pending_exit[code]
            else:
                del pending_exit[code]
        # 1b) 신규 진입(전일 신호)
        g = sig_by_day.get(days[i - 1]) if i > 0 else None
        if g is not None:
            for _, s in g.iterrows():
                if len(pos) >= b["max_positions"]:
                    break
                code = s["code"]
                if code in pos or code not in col:
                    continue
                j = col[code]
                if np.isnan(o[i, j]) or o[i, j] <= 0:
                    continue
                entry = o[i, j] * (1 + slip)
                stop = entry * (1 - stop_pct)
                ll = s.get("last_low", np.nan)
                if b["use_contraction_low"] and pd.notna(ll) and ll < entry:
                    stop = max(stop, ll)                    # 더 가까운(높은) 쪽
                stop = max(stop, entry * (1 - b["stop_max"]))
                R = entry - stop
                if R <= 0:
                    continue
                w = min(b["risk_per_trade"] / (R / entry), b["max_weight"])
                alloc = min(prev_equity * w, cash)
                if alloc < prev_equity * 0.01:
                    continue
                cost_ps = entry * (1 + comm)
                sh = alloc / cost_ps
                cash -= sh * cost_ps
                meta = {k: s[k] for k in s.index if k not in ("date", "code", "priority")}
                meta["signal_date"] = s["date"]
                pos[code] = {"shares": sh, "entry": entry, "entry_cost_ps": cost_ps, "stop": stop, "stop0": stop,
                             "R": R, "i_entry": i, "max_c": entry, "partial_done": False,
                             "weight0": sh * cost_ps / prev_equity, "meta": meta}
        # 2) 장중 손절
        for code in list(pos):
            j = col[code]; p = pos[code]
            if np.isnan(l[i, j]):
                continue
            if o[i, j] <= p["stop"]:
                close_pos(code, i, o[i, j], "stop_gap")
            elif l[i, j] <= p["stop"]:
                close_pos(code, i, p["stop"], "stop" if p["stop"] < p["entry"] else "trail_stop")
            elif pt["enabled"] and not p["partial_done"] and i - p["i_entry"] <= pt["within_days"] \
                    and h[i, j] >= p["entry"] * (1 + pt["gain"]):
                close_pos(code, i, p["entry"] * (1 + pt["gain"]), "partial_take", pt["fraction"])
        # 3) 종가 처리
        for code in list(pos):
            j = col[code]; p = pos[code]
            cc = c[i, j]
            if last_i.get(code) == i and i < len(days) - 1:
                close_pos(code, i, c_ff[i, j], "delisted"); continue
            if np.isnan(cc):
                continue
            p["max_c"] = max(p["max_c"], cc)
            if b["trail_3r_to_1r"] and p["max_c"] >= p["entry"] + 3 * p["R"]:
                p["stop"] = max(p["stop"], p["entry"] + p["R"])
            elif b["trail_2r_to_breakeven"] and p["max_c"] >= p["entry"] + 2 * p["R"]:
                p["stop"] = max(p["stop"], p["entry"])
            if cc < m50[i, j] and v[i, j] >= avg50[i, j] * b["ma50_exit_vol"]:
                pending_exit[code] = "ma50_break"
            elif cc < m200[i, j]:
                pending_exit[code] = "below_ma200"          # 4단계 진입 — 보유하지 않는다
        equity = cash + sum(p["shares"] * c_ff[i, col[k]] for k, p in pos.items())
        eq.append(equity); nh.append(len(pos))
        prev_equity = equity

    # 기간 말 미청산 포지션은 평가만(열린 거래로 기록)
    for code in list(pos):
        p = pos[code]; j = col[code]
        trades.append({**p["meta"], "code": code, "entry_date": days[p["i_entry"]], "entry_px": p["entry"],
                       "stop0": p["stop0"], "exit_date": pd.NaT, "exit_px": c_ff[d1 - 1, j], "reason": "open",
                       "frac": 1.0, "ret": c_ff[d1 - 1, j] / p["entry_cost_ps"] - 1,
                       "r_mult": (c_ff[d1 - 1, j] - p["entry"]) / p["R"], "hold_days": d1 - 1 - p["i_entry"],
                       "weight0": p["weight0"], "max_gain": p["max_c"] / p["entry"] - 1})
    idx = days[d0:d1]
    return {"label": label, "equity": pd.Series(eq, index=idx, name=label),
            "n_hold": pd.Series(nh, index=idx), "trades": pd.DataFrame(trades)}


# ───────────────────────── 성과 지표 ─────────────────────────
def perf(equity: pd.Series, trades: pd.DataFrame | None = None, bans=None) -> dict:
    r = equity.pct_change().dropna()
    yrs = len(r) / 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    dd = equity / equity.cummax() - 1
    out = {"CAGR": cagr, "Vol": vol, "MDD": dd.min(), "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan}
    if trades is not None and len(trades):
        t = trades[trades["reason"] != "open"]
        win, loss = t[t["ret"] > 0], t[t["ret"] <= 0]
        out.update({"Trades": len(t), "WinRate": len(win) / len(t) if len(t) else np.nan,
                    "AvgWin": win["ret"].mean(), "AvgLoss": loss["ret"].mean(),
                    "PayoffRatio": win["ret"].mean() / -loss["ret"].mean() if len(loss) and len(win) else np.nan,
                    "AvgHold": t["hold_days"].mean(),
                    "Turnover": (t["weight0"] * t["frac"]).sum() / yrs if yrs > 0 else np.nan})
    if bans:
        inb = pd.Series(False, index=r.index)
        for a, z in bans:
            inb |= (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(z))
        ann = lambda x: (1 + x).prod() ** (252 / len(x)) - 1 if len(x) else np.nan
        out["CAGR_shortban"] = ann(r[inb]); out["CAGR_ex_shortban"] = ann(r[~inb])
    return out


def yearly(equity: pd.Series) -> pd.Series:
    e = equity.resample("YE").last()
    first = equity.iloc[0]
    return e.pct_change().fillna(e.iloc[0] / first - 1)
