# -*- coding: utf-8 -*-
"""output/experiments.pkl → backtest_report.html, trades.csv, top_trades.csv, vcp_samples/*.png, tearsheet_D.html."""
from __future__ import annotations

import base64
import html
import io
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt
import data_loader as dl

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
CFG = dl.CFG
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
COLORS = ["#2a6fdb", "#e0473a", "#1f9d6b", "#8a56d6", "#e39a1c", "#5b6b7a", "#c2417e", "#1aa0b8"]


def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pct(x, d=1):
    return "" if pd.isna(x) else f"{x*100:.{d}f}%"


def num(x, d=2):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def table(df: pd.DataFrame, fmt: dict | None = None, cls="") -> str:
    fmt = fmt or {}
    h = [f'<table class="{cls}"><thead><tr><th></th>' + "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns) + "</tr></thead><tbody>"]
    for i, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            f = fmt.get(c)
            s = f(v) if f else (v if isinstance(v, str) else ("" if pd.isna(v) else f"{v}"))
            neg = isinstance(v, (int, float, np.floating)) and not pd.isna(v) and v < 0 and f in (pct,)
            cells.append(f'<td class="{"neg" if neg else ""}">{s}</td>')
        h.append(f"<tr><th>{html.escape(str(i))}</th>{''.join(cells)}</tr>")
    h.append("</tbody></table>")
    return "".join(h)


PERF_FMT = {"CAGR": pct, "Vol": pct, "MDD": pct, "Sharpe": num, "WinRate": pct, "AvgWin": pct, "AvgLoss": pct,
            "PayoffRatio": num, "AvgHold": lambda x: num(x, 0), "Turnover": lambda x: num(x, 1),
            "Trades": lambda x: "" if pd.isna(x) else f"{int(x)}", "CAGR_shortban": pct, "CAGR_ex_shortban": pct,
            "IS_CAGR": pct, "OOS_CAGR": pct, "IS_Sharpe": num, "OOS_Sharpe": num, "vs_KOSPI": pct}


def perf_table(R: dict, keys, bench: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    ise = pd.Timestamp(CFG["backtest"]["in_sample_end"])
    for k in keys:
        if k not in R:
            continue
        v = R[k]; e = v["equity"]
        p = bt.perf(e, v["trades"], CFG["backtest"]["short_ban"])
        ins, oos = e[e.index <= ise], e[e.index > ise]
        p["IS_CAGR"] = bt.perf(ins)["CAGR"] if len(ins) > 60 else np.nan
        p["OOS_CAGR"] = bt.perf(oos)["CAGR"] if len(oos) > 60 else np.nan
        p["IS_Sharpe"] = bt.perf(ins)["Sharpe"] if len(ins) > 60 else np.nan
        p["OOS_Sharpe"] = bt.perf(oos)["Sharpe"] if len(oos) > 60 else np.nan
        kb = bench["KOSPI"].reindex(e.index).ffill()
        p["vs_KOSPI"] = p["CAGR"] - bt.perf(kb)["CAGR"]
        rows[v["label"]] = p
    cols = ["CAGR", "vs_KOSPI", "Vol", "MDD", "Sharpe", "Trades", "WinRate", "AvgWin", "AvgLoss", "PayoffRatio",
            "AvgHold", "Turnover", "IS_CAGR", "OOS_CAGR", "IS_Sharpe", "OOS_Sharpe", "CAGR_shortban", "CAGR_ex_shortban"]
    return pd.DataFrame(rows).T.reindex(columns=cols)


def bench_table(bench: pd.DataFrame, start) -> pd.DataFrame:
    rows = {}
    for c in bench.columns:
        s = bench[c][bench.index >= pd.Timestamp(start)].dropna()
        rows[c] = bt.perf(s, None, CFG["backtest"]["short_ban"])
    return pd.DataFrame(rows).T[["CAGR", "Vol", "MDD", "Sharpe", "CAGR_shortban", "CAGR_ex_shortban"]]


def equity_chart(R, keys, bench, title):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    start = min(R[k]["equity"].index[0] for k in keys if k in R)
    for i, k in enumerate([k for k in keys if k in R]):
        e = R[k]["equity"]; ax.plot(e.index, e / e.iloc[0], label=R[k]["label"], color=COLORS[i % len(COLORS)], lw=1.4)
    for c, ls in (("KOSPI", "--"), ("KOSPI 동일가중", ":")):
        b = bench[c][bench.index >= start].dropna(); ax.plot(b.index, b / b.iloc[0], label=c, color="#333", ls=ls, lw=1.1)
    ax.set_yscale("log"); ax.set_title(title); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
    return fig_b64(fig)


def yearly_table(R, keys, bench):
    cols = {}
    for k in keys:
        if k in R:
            cols[R[k]["label"]] = bt.yearly(R[k]["equity"])
    start = min(R[k]["equity"].index[0] for k in keys if k in R)
    for c in ("KOSPI", "KOSPI 동일가중"):
        b = bench[c][bench.index >= start].dropna(); cols[c] = bt.yearly(b)
    t = pd.DataFrame(cols); t.index = t.index.year
    return t


# ───────────────────────── 신호 단위 이벤트 스터디 ─────────────────────────
def forward_excess(sig: pd.DataFrame, px, uni, horizons=(20, 60, 120)) -> pd.DataFrame:
    """신호 다음날 시가 매수 → t+N 종가. 같은 기간 유니버스 동일가중 대비 초과수익."""
    O, C = px["adj_open"], px["adj_close"].ffill()
    days = C.index
    r1 = C.pct_change(fill_method=None).where(uni.shift(1).fillna(False).astype(bool))
    ew = (1 + r1.mean(axis=1).fillna(0)).cumprod()
    pos = days.get_indexer(sig["date"])
    col = C.columns.get_indexer(sig["code"])
    out = sig.copy()
    for n in horizons:
        ok = (pos + n < len(days)) & (col >= 0)
        v = np.full(len(sig), np.nan)
        pi, ci = pos[ok], col[ok]
        entry = O.values[pi + 1, ci]
        ex = C.values[pi + n, ci]
        v[ok] = ex / entry - 1 - (ew.values[pi + n] / ew.values[pi] - 1)
        out[f"x{n}"] = v
    return out


def event_summary(fx: pd.DataFrame, by: str, label_map=None) -> pd.DataFrame:
    g = fx.groupby(by)
    t = pd.DataFrame({"신호 수": g.size(), "20일 초과": g["x20"].mean(), "60일 초과": g["x60"].mean(),
                      "120일 초과": g["x120"].mean(), "60일 승률": g["x60"].apply(lambda s: (s > 0).mean())})
    if label_map:
        t.index = [label_map.get(i, i) for i in t.index]
    return t


EV_FMT = {"신호 수": lambda x: f"{int(x)}", "20일 초과": pct, "60일 초과": pct, "120일 초과": pct, "60일 승률": pct}


# ───────────────────────── VCP 샘플 차트 ─────────────────────────
def vcp_samples(sig: pd.DataFrame, px, names, n=20, seed=7):
    d = OUT / "vcp_samples"; d.mkdir(exist_ok=True)
    for f in d.glob("*.png"):
        f.unlink()
    s = sig[sig["kind"] == "vcp"]
    s = s.sample(min(n, len(s)), random_state=seed).sort_values("date")
    days = px["adj_close"].index
    files = []
    for _, r in s.iterrows():
        c = r["code"]; t = days.get_loc(r["date"]); b0 = days.get_loc(r["base_start"])
        a, z = max(0, b0 - 40), min(len(days) - 1, t + 40)
        sl = slice(a, z + 1)
        hi, lo, cl, vo = (px[k][c].iloc[sl] for k in ("adj_high", "adj_low", "adj_close", "adj_volume"))
        fig, (ax, av) = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax.vlines(hi.index, lo.values, hi.values, color="#999", lw=0.8)
        ax.plot(cl.index, cl.values, color="#2a6fdb", lw=1.2)
        ax.axhline(r["pivot"], color="#e0473a", ls="--", lw=1, label=f"피벗 {r['pivot']:,.0f}")
        ax.axhline(r["last_low"], color="#1f9d6b", ls=":", lw=1, label=f"마지막 수축 저점 {r['last_low']:,.0f}")
        ax.axvspan(r["base_start"], r["date"], color="#f4c542", alpha=.12, label="베이스")
        ax.axvline(r["date"], color="#e0473a", lw=0.8)
        dep = "→".join(f"{x*100:.0f}%" for x in r["depths"])
        ax.set_title(f"{names.get(c, c)}({c}) {r['date'].date()}  수축 {dep}  거래량 건조 {r['dryup']:.2f}  돌파 거래량 ×{r['vol_ratio']:.1f}", fontsize=9)
        ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=.3)
        av.bar(vo.index, vo.values, color=["#e0473a" if x == r["date"] else "#bbb" for x in vo.index], width=1)
        av.grid(alpha=.3)
        fn = d / f"{r['date'].strftime('%Y%m%d')}_{c}.png"
        fig.savefig(fn, dpi=100, bbox_inches="tight"); plt.close(fig)
        files.append(fn)
    return files


# ───────────────────────── 주도주 타임라인 (질문 7) ─────────────────────────
def leaders(px, uni, sector, names, start="2023-01-02", n=10, per_sector=2, min_cap=5e11):
    C = px["adj_close"]
    d0 = C.index[C.index.searchsorted(pd.Timestamp(start))]
    elig = uni.loc[d0] & (px["mktcap"].loc[d0] >= min_cap)
    r = (C.iloc[-1] / C.loc[d0] - 1)[elig[elig].index].dropna().sort_values(ascending=False)
    pick, cnt = [], {}
    for c in r.index:
        s = sector.get(c) or "?"
        if cnt.get(s, 0) >= per_sector:
            continue
        pick.append(c); cnt[s] = cnt.get(s, 0) + 1
        if len(pick) >= n:
            break
    return pick, r


def leader_timeline(codes, px, trades_by_exp: dict, sigs: dict, names, start="2023-01-02"):
    C = px["adj_close"]
    rows, imgs = [], []
    fig, axes = plt.subplots((len(codes) + 1) // 2, 2, figsize=(12, 2.4 * ((len(codes) + 1) // 2)), squeeze=False)
    for k, c in enumerate(codes):
        s = C[c][C.index >= pd.Timestamp(start)].dropna()
        ax = axes[k // 2][k % 2]
        ax.plot(s.index, s.values, color="#333", lw=1)
        ax.set_title(f"{names.get(c, c)} ({c})  {s.iloc[-1]/s.iloc[0]-1:+.0%}", fontsize=9)
        for (lab, t), colr in zip(trades_by_exp.items(), ("#2a6fdb", "#e0473a")):
            tt = t[(t.code == c) & (t.entry_date >= pd.Timestamp(start))]
            for _, x in tt.iterrows():
                ex = x["exit_date"] if pd.notna(x["exit_date"]) else s.index[-1]
                ax.axvspan(x["entry_date"], ex, color=colr, alpha=.15)
                rows.append({"종목": names.get(c, c), "코드": c, "실험": lab, "진입일": x["entry_date"].date(),
                             "청산일": (x["exit_date"].date() if pd.notna(x["exit_date"]) else "보유중"),
                             "수익률": x["ret"], "청산 사유": x["reason"]})
        for lab, sg in sigs.items():
            ss = sg[(sg.code == c) & (sg.date >= pd.Timestamp(start))]
            ax.scatter(ss["date"], C[c].reindex(ss["date"]).values, s=10, color="#1f9d6b", zorder=3)
        ax.grid(alpha=.3); ax.tick_params(labelsize=7)
    for k in range(len(codes), axes.size):
        axes.flat[k].axis("off")
    fig.suptitle("주도주 타임라인 — 파랑: D 보유 구간, 빨강: A 보유 구간, 초록 점: VCP 신호(펀더멘털 무관)", fontsize=10)
    fig.tight_layout()
    return pd.DataFrame(rows), fig_b64(fig)


def first_dates(codes, tt_pass, sig_all, start="2023-01-02"):
    rows = []
    for c in codes:
        s = tt_pass[c][tt_pass.index >= pd.Timestamp(start)]
        ft = s.index[s.fillna(False).astype(bool)].min() if s.any() else pd.NaT
        vs = sig_all[(sig_all.code == c) & (sig_all.date >= pd.Timestamp(start))]
        fv = vs["date"].min() if len(vs) else pd.NaT
        fvf = vs[vs.fund_pass]["date"].min() if len(vs) else pd.NaT
        rows.append({"code": c, "첫 추세템플릿 통과": ft, "첫 VCP 신호": fv, "첫 VCP+펀더멘털 신호": fvf,
                     "VCP 신호 수": len(vs), "그중 펀더멘털 통과": int(vs["fund_pass"].sum()) if len(vs) else 0})
    return pd.DataFrame(rows)


# ───────────────────────── 티어시트 ─────────────────────────
def tearsheet(equity: pd.Series, bench: pd.Series, title: str, fn: Path):
    import tossquant as qs
    plt.rcParams["font.family"] = "Malgun Gothic"        # tossquant 가 import 때 Pretendard 로 바꾸므로 덮어쓴다
    plt.rcParams["axes.unicode_minus"] = False
    r = equity.pct_change().dropna()
    b = bench.reindex(equity.index).ffill().pct_change().dropna()
    r.index = r.index.tz_localize(None); b.index = b.index.tz_localize(None)
    qs.reports.html(r, benchmark=b, title=title, output=str(fn), periods_per_year=252)   # 일별 수익률 → 252
    t = fn.read_text(encoding="utf-8").replace('<body onload="save()">', "<body>")
    fn.write_text(t, encoding="utf-8")


# ───────────────────────── 조립 ─────────────────────────
CSS = """
:root{--bg:#fff;--fg:#1d2330;--mut:#667085;--line:#e4e7ec;--card:#f8f9fb;--neg:#c2362b;--acc:#2a6fdb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#12151c;--fg:#e6e8ee;--mut:#98a2b3;--line:#2a303c;--card:#1a1f29;--neg:#ff7b70;--acc:#6ea0ff}}
body{background:var(--bg);color:var(--fg);font:14px/1.6 'Malgun Gothic','Apple SD Gothic Neo',sans-serif;margin:0}
main{max-width:1200px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:24px;margin:8px 0 4px}h2{font-size:19px;margin:36px 0 10px;border-bottom:2px solid var(--line);padding-bottom:6px}
h3{font-size:15px;margin:22px 0 8px}.mut{color:var(--mut)}
.wrap{overflow-x:auto}table{border-collapse:collapse;font-size:12.5px;margin:6px 0 14px;white-space:nowrap}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}thead th{background:var(--card);position:sticky;top:0}
tbody th{text-align:left;font-weight:600}td.neg{color:var(--neg)}
img{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:10px 0}
.concl h3{margin-top:14px}.concl p,.concl li{margin:4px 0}
nav a{margin-right:12px;color:var(--acc)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:10px}
"""


def md_to_html(md: str) -> str:
    """conclusions.md 용 최소 마크다운(### 제목, - 목록, **굵게**, 문단)."""
    import re
    out, in_ul = [], False
    for line in md.splitlines():
        s = html.escape(line.rstrip())
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        if s.startswith("### "):
            if in_ul:
                out.append("</ul>"); in_ul = False
            out.append(f"<h3>{s[4:]}</h3>")
        elif s.startswith("- "):
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{s[2:]}</li>")
        elif s.strip():
            if in_ul:
                out.append("</ul>"); in_ul = False
            out.append(f"<p>{s}</p>")
    if in_ul:
        out.append("</ul>")
    return "\n".join(out)


def _d(x):
    return "" if pd.isna(x) else str(pd.Timestamp(x).date())


def _depths(d):
    return "→".join(f"{x*100:.0f}%" for x in d) if isinstance(d, (list, np.ndarray)) else ""


def build():
    import trend
    res = pickle.load(open(OUT / "experiments.pkl", "rb"))
    R, bench, reg = res["R"], res["bench"], res["reg"]
    px = dl.load_panel()
    names = dl.load_names(); sector = dl.load_sector()
    uni = dl.universe_mask(px, CFG, names, ["KOSPI"])
    S = []
    add = S.append
    ypct = lambda t: {c: pct for c in t.columns}

    # 1. 결론
    cf = OUT / "conclusions.md"
    body = md_to_html(cf.read_text(encoding="utf-8")) if cf.exists() else "<p>(conclusions.md 작성 전)</p>"
    add('<h2 id="concl">결론 — 4장 질문에 대한 답</h2>')
    add(f'<div class="card concl">{body}</div>')

    # 2. 전체 기간
    full = ["A_full", "A_reg_full", "V_full", "V_reg_full"]
    add('<h2 id="full">전체 기간 2011~ — 추세 전용 전략</h2>')
    add('<p class="mut">DART 분기재무가 2016Q1부터라 펀더멘털 없는 전략만 2011년부터 돌렸다.</p>')
    add(f'<div class="wrap">{table(perf_table(R, full, bench), PERF_FMT)}</div>')
    add(f'<p class="mut">벤치마크(2011~)</p><div class="wrap">{table(bench_table(bench, CFG["backtest"]["start"]), PERF_FMT)}</div>')
    add(f'<img src="{equity_chart(R, full, bench, "2011~ 누적(로그)")}">')
    yt = yearly_table(R, full, bench)
    add(f'<h3>연도별 수익률</h3><div class="wrap">{table(yt, ypct(yt))}</div>')

    # 3. 공통 기간 A~H
    core = ["A", "B", "V", "C", "D", "E", "F", "G", "H5", "H7", "H8", "H10", "H12", "H15", "D_ew", "D_breadth", "D_det", "D_strict", "D_pt"]
    add('<h2 id="abl">실험 A~H — 공통 기간 2016-06~</h2>')
    add('<p class="mut">A 추세템플릿+50일 신고가 돌파 · B +펀더멘털 · C0 VCP만 · C B+VCP · D C+국면(KOSPI&gt;상승 200일선) · '
        'E D+촉매 우선순위 · F D+코스닥 · G D−경기민감 · H 손절폭 · D′ 동일가중 국면 · D″ 브레드스 국면 · '
        'D‴ 유진 리서치 디텍터 국면 · 흑자전환 불인정 · 분할매도. 표본 내 ~2019 / 표본 외 2020~.</p>')
    add(f'<div class="wrap">{table(perf_table(R, core, bench), PERF_FMT)}</div>')
    add(f'<p class="mut">벤치마크(2016-06~)</p><div class="wrap">{table(bench_table(bench, "2016-06-01"), PERF_FMT)}</div>')
    add(f'<img src="{equity_chart(R, ["A", "B", "V", "C", "D", "F"], bench, "2016-06~ 누적(로그)")}">')
    yt = yearly_table(R, ["A", "B", "C", "D", "E", "F", "G"], bench)
    add(f'<h3>연도별 수익률</h3><div class="wrap">{table(yt, ypct(yt))}</div>')
    add('<div class="card mut">공매도 금지 구간: 2020-03-16~2021-05-02, 2023-11-06~2025-03-30. '
        'CAGR_shortban / CAGR_ex_shortban 은 해당 구간 안/밖 일수익률만 모아 연환산한 값.</div>')

    # 4. 신호 단위 이벤트 스터디
    add('<h2 id="ev">신호 단위 이벤트 스터디 — 필터 자체의 가치</h2>')
    add('<p class="mut">포트폴리오 슬롯(10종목) 제약 없이 모든 신호를 다음날 시가에 샀다고 보고, '
        't+N 종가까지 코스피 유니버스 동일가중 대비 초과수익.</p>')
    fa = forward_excess(res["sig_A"], px, uni)
    fv = forward_excess(res["sig_vcp"], px, uni)
    fa16 = fa[fa.date >= "2016-06-01"].copy(); fv16 = fv[fv.date >= "2016-06-01"].copy()
    add("<h3>질문 1·2 — 추세 돌파 신호: 펀더멘털 통과 여부(2016-06~)</h3>")
    add(table(event_summary(fa16, "fund_pass", {True: "펀더멘털 통과", False: "미통과(테마·수급성 포함)"}), EV_FMT))
    add("<h3>추세 돌파 신호 전체(2011~) 연도별</h3>")
    fa["year"] = fa.date.dt.year
    add(f'<div class="wrap">{table(event_summary(fa, "year"), EV_FMT)}</div>')
    add("<h3>질문 3 — 진입 방식: 단순 돌파 vs VCP 피벗(2016-06~, 펀더멘털 통과만)</h3>")
    cmp = pd.concat([fa16[fa16.fund_pass].assign(k="단순 50일 신고가 돌파"),
                     fv16[fv16.fund_pass].assign(k=lambda d: d["kind"].map({"vcp": "VCP 피벗", "pp": "파워 플레이"}))])
    add(table(event_summary(cmp, "k"), EV_FMT))
    add("<h3>질문 3 보조 — 펀더멘털 무관 전체(2011~)</h3>")
    cmp2 = pd.concat([fa.assign(k="단순 50일 신고가 돌파"), fv.assign(k=lambda d: d["kind"].map({"vcp": "VCP 피벗", "pp": "파워 플레이"}))])
    add(table(event_summary(cmp2, "k"), EV_FMT))
    grp = lambda d: np.where(d.is_cyclical, "경기민감", "비경기민감") + np.where(d.code33, " · 코드33", np.where(d.fund_pass, " · 펀더멘털 통과", " · 미통과"))
    add("<h3>질문 5 — 경기민감주 vs 성장주: 코드 33·펀더멘털 신호(2016-06~)</h3>")
    add("<p class='mut'>추세 돌파 신호 기준(표본이 크다)</p>" + table(event_summary(fa16.assign(grp=grp), "grp"), EV_FMT))
    add("<p class='mut'>VCP 신호 기준</p>" + table(event_summary(fv16.assign(grp=grp), "grp"), EV_FMT))
    add("<h3>촉매 점수 상·하위(VCP, 2016-06~)</h3>")
    if fv16["cat_score"].notna().sum() >= 3:
        fv16["cat_q"] = pd.qcut(fv16["cat_score"].rank(method="first"), 3, labels=["하위 1/3", "중간", "상위 1/3"])
        add(table(event_summary(fv16, "cat_q"), EV_FMT))
        nh = fv16.assign(nh=fv16.cat_new_high.map({1.0: "60일내 52주 신고가", 0.0: "신고가 없음"}))
        add(table(event_summary(nh, "nh"), EV_FMT))
        th = fv16.assign(th=np.where(fv16.cat_theme.notna(), "Core 테마 소속", "테마 없음"))
        add(table(event_summary(th, "th"), EV_FMT))

    # 5. 손절폭(질문 6)
    add('<h2 id="stop">질문 6 — 손절폭 민감도</h2>')
    hk = [f"H{x}" for x in (5, 7, 8, 10, 12, 15)] + [f"HB{x}" for x in (5, 7, 8, 10, 12, 15)]
    add("<p class='mut'>D 는 거래가 적어(수십 건) 경로 의존이 크다. 같은 손절 규칙을 표본이 큰 B(추세+펀더멘털+단순 돌파)에도 적용했다. "
        "10% 초과는 책의 상한(최대 10%)을 넘는 확장 실험이다.</p>")
    add(f'<div class="wrap">{table(perf_table(R, hk, bench), PERF_FMT)}</div>')
    tD = R["D"]["trades"]
    if len(tD):
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.hist(tD["ret"].clip(-0.15, 1.0), bins=60, color="#2a6fdb")
        ax.set_title("실험 D 거래별 수익률 분포(상한 100%)"); ax.grid(alpha=.3)
        add(f'<img src="{fig_b64(fig)}">')
        g = tD.groupby("reason").agg(건수=("ret", "size"), 평균=("ret", "mean"), 평균보유일=("hold_days", "mean"))
        add(table(g, {"건수": lambda x: f"{int(x)}", "평균": pct, "평균보유일": lambda x: num(x, 0)}))
    # 진입 후 역행폭(MAE) 분포 — 손절폭 적정성 보조
    fvm = mae_table(res["sig_vcp"][res["sig_vcp"].fund_pass], px)
    if len(fvm):
        add("<p class='mut'>VCP+펀더멘털 신호의 진입 후 20일 최대 역행폭(MAE) 분포 — 최종 60일 성과 상위/하위 절반</p>")
        add(table(fvm, {c: pct for c in fvm.columns}))

    # 6. 국면
    add('<h2 id="reg">질문 4 — 시장 국면</h2>')
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    k = reg["kospi"]
    a1.plot(k.index, k, color="#333", lw=1)
    a1.fill_between(k.index, k.min(), k.max(), where=reg["mkt_ok"].values, color="#1f9d6b", alpha=.12, label="mkt_ok")
    a1.set_yscale("log"); a1.legend(fontsize=8); a1.grid(alpha=.3); a1.set_title("KOSPI 와 국면 필터(초록 = 신규 진입 허용)")
    a2.plot(reg.index, reg["breadth"], color="#8a56d6", lw=1)
    a2.axhline(CFG["regime"]["breadth_min"], color="#e0473a", ls="--", lw=1)
    a2.set_title("브레드스: 추세 템플릿 통과 비율"); a2.grid(alpha=.3)
    add(f'<img src="{fig_b64(fig)}">')
    yt = yearly_table(R, ["A_full", "A_reg_full", "V_full", "V_reg_full"], bench)
    sel = yt.loc[[y for y in yt.index if y <= 2016 or y in (2018, 2022)]]
    add("<h3>박스피(2011~2016)·약세장(2018, 2022) 연도별</h3>" + f'<div class="wrap">{table(sel, ypct(sel))}</div>')

    # 7. 주도주(질문 7)
    add('<h2 id="lead">질문 7 — 2023~ 주도주를 언제 잡고 언제 놓쳤나</h2>')
    tt = trend.compute(px, uni, CFG)["tt_pass"]
    codes, rr = leaders(px, uni, sector, names)
    add("<p class='mut'>선정: 2023-01-02 유니버스(코스피, 시총 5,000억 이상) 중 최신일까지 수익률 상위, 업종당 최대 2종목. "
        "기억이 아니라 데이터로 고른 목록이다.</p>")
    lst = pd.DataFrame({"종목": [names.get(c, c) for c in codes], "업종": [sector.get(c) for c in codes],
                        "2023~ 수익률": [rr[c] for c in codes]}, index=codes)
    add(table(lst, {"2023~ 수익률": pct}))
    fd = first_dates(codes, tt, res["sig_vcp"]).set_index("code")
    fd.index = [names.get(c, c) for c in fd.index]
    add(table(fd, {k_: _d for k_ in ("첫 추세템플릿 통과", "첫 VCP 신호", "첫 VCP+펀더멘털 신호")}))
    tl, img = leader_timeline(codes, px, {"D": R["D"]["trades"], "A": R["A"]["trades"]}, {"vcp": res["sig_vcp"]}, names)
    add(f'<img src="{img}">')
    if len(tl):
        add(f'<div class="wrap">{table(tl.set_index("종목"), {"수익률": pct})}</div>')
    lead_tbl = tl

    # 8. 상위 거래 20
    add('<h2 id="top">상위 수익 거래 20 — 책의 "초수익 종목" 특징과 비교</h2>')
    add("<p class='mut'>D 는 거래가 24건뿐이라 표본이 큰 '추세+펀더멘털+돌파, 손절 8%'(HB8) 거래에서 뽑았다.</p>")
    tB = R["HB8"]["trades"]
    tB = tB[tB["reason"] != "open"] if len(tB) else tB
    if len(tB):
        top = tB.sort_values("ret", ascending=False).head(20)
        cols = ["name", "sector", "entry_date", "exit_date", "ret", "hold_days", "rs", "eps_yoy_q0", "sales_yoy_q0",
                "code33", "kind", "n_contr", "depths", "base_len", "is_cyclical", "reason"]
        top = top[[c for c in cols if c in top.columns]].copy()
        if "depths" in top:
            top["depths"] = top["depths"].apply(_depths)
        top.to_csv(OUT / "top_trades.csv", index=False, encoding="utf-8-sig")
        add(f'<div class="wrap">{table(top.reset_index(drop=True), {"ret": pct, "eps_yoy_q0": pct, "sales_yoy_q0": pct, "entry_date": _d, "exit_date": _d})}</div>')
        m = ["rs", "eps_yoy_q0", "sales_yoy_q0", "base_len", "n_contr"]
        cmpt = pd.DataFrame({"상위 20": top.reindex(columns=m).median(numeric_only=True),
                             "HB8 전체": tB.reindex(columns=m).median(numeric_only=True)})
        cmpt.loc["경기민감 비율"] = [top["is_cyclical"].mean(), tB["is_cyclical"].mean()]
        cmpt.loc["흑자전환(EPS NaN) 비율"] = [top["eps_yoy_q0"].isna().mean(), tB["eps_yoy_q0"].isna().mean()]
        add("<p class='mut'>진입 시점 특성 중앙값</p>" + table(cmpt, {"상위 20": num, "HB8 전체": num}))

    # 9. VCP 샘플
    add('<h2 id="vcp">VCP 탐지 샘플 20 — 눈 검수용</h2>')
    files = vcp_samples(res["sig_vcp"], px, names)
    add('<div class="grid">' + "".join(f'<img src="vcp_samples/{f.name}">' for f in files) + "</div>")

    # 10. 데이터·한계
    add('<h2 id="data">데이터 커버리지와 비활성 요소</h2>')
    cov = (dl.DATA / "COVERAGE.md").read_text(encoding="utf-8")
    add(f'<div class="card"><pre style="white-space:pre-wrap;font-size:12px">{html.escape(cov)}</pre></div>')

    tearsheet(R["D"]["equity"], bench["KOSPI"], "SEPA 코스피 — 실험 D (유진 리서치)", OUT / "tearsheet_D.html")
    tearsheet(R["A_full"]["equity"], bench["KOSPI"], "SEPA 코스피 — 실험 A 2011~ (유진 리서치)", OUT / "tearsheet_A_full.html")
    tearsheet(R["HB8"]["equity"], bench["KOSPI"], "SEPA 코스피 — 추세+펀더멘털+돌파, 손절 8% (유진 리서치)", OUT / "tearsheet_B8.html")

    allt = []
    for key, v in R.items():
        t = v["trades"].copy()
        if len(t):
            t.insert(0, "experiment", key)
            if "depths" in t:
                t["depths"] = t["depths"].apply(_depths)
            allt.append(t)
    pd.concat(allt).to_csv(OUT / "trades.csv", index=False, encoding="utf-8-sig")
    if len(lead_tbl):
        lead_tbl.to_csv(OUT / "leaders_timeline.csv", index=False, encoding="utf-8-sig")

    last = px["adj_close"].index[-1].date()
    head = (f'<h1>초수익 성장주(SEPA) 코스피 백테스트</h1><p class="mut">유진 리서치 · 데이터 {CFG["data"]["start"]} ~ {last} · '
            '가격 한투 Open API(수정주가, 상장폐지 포함) · 재무 DART(포인트인타임)</p>'
            '<nav><a href="#concl">결론</a><a href="#full">2011~</a><a href="#abl">실험 A~H</a><a href="#ev">이벤트 스터디</a>'
            '<a href="#stop">손절폭</a><a href="#reg">국면</a><a href="#lead">주도주</a><a href="#top">상위 거래</a>'
            '<a href="#vcp">VCP 샘플</a><a href="#data">데이터</a><a href="tearsheet_D.html">티어시트 D</a>'
            '<a href="tearsheet_A_full.html">티어시트 A</a><a href="tearsheet_B8.html">티어시트 B(8%)</a></nav>')
    doc = ('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>SEPA 코스피 백테스트</title><style>{CSS}</style></head><body><main>{head}{"".join(S)}</main></body></html>')
    (OUT / "backtest_report.html").write_text(doc, encoding="utf-8")
    print("report →", OUT / "backtest_report.html")


def mae_table(sig, px, n=20):
    """신호 다음날 시가 진입 후 n일 최저가 기준 최대 역행폭. 최종 60일 수익 상/하위 절반으로 나눠 분위수."""
    O, L, C = px["adj_open"], px["adj_low"], px["adj_close"].ffill()
    days = C.index
    rows = []
    for d, c in zip(sig["date"], sig["code"]):
        i = days.get_loc(d)
        if i + 61 >= len(days) or c not in C.columns:
            continue
        e = O[c].iloc[i + 1]
        if not e or np.isnan(e):
            continue
        rows.append({"mae": L[c].iloc[i + 1:i + 1 + n].min() / e - 1, "r60": C[c].iloc[i + 60] / e - 1})
    if not rows:
        return pd.DataFrame()
    m = pd.DataFrame(rows)
    m["grp"] = np.where(m["r60"] >= m["r60"].median(), "60일 성과 상위 절반", "하위 절반")
    q = m.groupby("grp")["mae"].quantile([0.25, 0.5, 0.75]).unstack()
    q.columns = ["MAE 25%", "MAE 중앙값", "MAE 75%"]
    return q


if __name__ == "__main__":
    build()
