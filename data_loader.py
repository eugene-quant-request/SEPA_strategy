# -*- coding: utf-8 -*-
"""데이터 수집·캐시·패널 구성.

가격은 pykrx '날짜별 전종목' 조회 하나로 만든다.
- pykrx 종목별 수정주가(adjusted=True)는 오늘부터 약 3,000영업일(2014-07~)까지만 내려준다. 2010~2014 불가.
- 날짜별 전종목 조회는 수정주가가 아니지만 '등락률'은 KRX 가 권리락·분할을 반영한 기준가 대비로 계산한다.
  → 등락률을 누적해 수정종가 지수를 만들고, (수정종가/원종가) 비율을 OHLC 에 곱해 수정 OHLC 를 만든다.
  이 방식은 상장폐지 종목도 자동으로 포함하므로 생존편향이 없다. 검증은 validate_adjustment() 참고.

사용:  python data_loader.py fetch-prices | build | fetch-index | fetch-dart | fetch-sector | fetch-names | fetch-flows
"""
from __future__ import annotations

import glob
import io
import json
import os
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
DATA = ROOT / CFG["data"]["dir"]
RAW = DATA / "raw_daily"
DATA.mkdir(exist_ok=True)

INDEX_CODES = {"KOSPI": "1001", "KOSDAQ": "2001", "KOSPI200": "1028"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _krx():
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):   # 로그인 메시지 숨김
        from pykrx import stock
    return stock


def _retry(fn, *a, tries=6, **k):
    for i in range(tries):
        try:
            return fn(*a, **k)
        except Exception as e:  # 네트워크·세션 만료
            if i == tries - 1:
                raise
            log("retry", fn.__name__, a[:2], repr(e)[:120])
            time.sleep(10 * (i + 1))


def end_date() -> str:
    """수집 종료일. 한투 일봉의 당일 종가는 시간외 단일가(16:00~18:00)까지 바뀌므로 18시 전이면 전날까지만 받는다."""
    if CFG["data"]["end"]:
        return CFG["data"]["end"]
    now = pd.Timestamp.now()
    d = now.normalize() if now.hour >= 18 else now.normalize() - pd.Timedelta(days=1)
    return d.strftime("%Y-%m-%d")


# ───────────────────────── 지수 / 영업일 ─────────────────────────
def fetch_index():
    stock = _krx()
    s, e = CFG["data"]["start"].replace("-", ""), end_date().replace("-", "")
    out = {}
    for name, code in INDEX_CODES.items():
        parts = []
        for y0 in range(int(s[:4]), int(e[:4]) + 1, 4):      # 긴 구간은 잘라서
            a = max(s, f"{y0}0101"); b = min(e, f"{y0+3}1231")
            parts.append(_retry(stock.get_index_ohlcv, a, b, code))
        df = pd.concat(parts)
        df = df[~df.index.duplicated()].sort_index()
        out[name] = df["종가"].astype(float)
        log("index", name, len(df), df.index.min().date(), df.index.max().date())
    pd.DataFrame(out).to_parquet(DATA / "index_close.parquet")


def trading_days() -> pd.DatetimeIndex:
    return pd.read_parquet(DATA / "index_close.parquet")["KOSPI"].dropna().index


# ───────────────────────── 날짜별 전종목 원시 시세 ─────────────────────────
def fetch_prices(markets=None):
    """월 단위 parquet 캐시(raw_daily/{market}/{YYYYMM}.parquet). 이미 받은 달은 건너뛴다(진행 중인 달은 다시)."""
    stock = _krx()
    markets = markets or CFG["data"]["markets"]
    days = trading_days()
    this_month = days.max().strftime("%Y%m")
    for mkt in markets:
        (RAW / mkt).mkdir(parents=True, exist_ok=True)
        for ym, ds in pd.Series(days, index=days).groupby(days.strftime("%Y%m")):
            f = RAW / mkt / f"{ym}.parquet"
            if f.exists() and ym != this_month:
                continue
            frames = []
            for d in ds:
                df = _retry(stock.get_market_ohlcv, d.strftime("%Y%m%d"), market=mkt)
                if df is None or df.empty:
                    log("EMPTY", mkt, d.date()); continue
                df = df.reset_index().rename(columns={"티커": "code"})
                df["date"] = d
                frames.append(df)
            m = pd.concat(frames, ignore_index=True).rename(columns={
                "시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume",
                "거래대금": "value", "등락률": "chg", "시가총액": "mktcap"})
            m["market"] = mkt
            m.to_parquet(f)
            log("prices", mkt, ym, len(ds), "days", m["code"].nunique(), "codes")


# ───────────────────────── 한투(KIS) Open API 경로 — 기본 가격 소스 ─────────────────────────
# KRX Data Marketplace 가 pykrx 대량 조회를 약관 위반(자동화 수집)으로 보고 IP 를 제한했다(2026-09-29).
# 그래서 가격은 한투 공식 API 로 받는다. 일봉(FHKST03010100, 수정주가)은 2009년까지, 상장폐지 종목도 내려준다.
# 종목 기본정보(CTPF1002R)로 시장·증권그룹·보통주 여부·상장/폐지일·업종·상장주식수를 얻는다.
class _KIS:
    def __init__(self, rps=15):
        sys.path.insert(0, str(ROOT.parent / "kis_api"))
        from kis_client import KISClient
        self.c = KISClient()
        import threading
        self.local = threading.local()
        self.lock = threading.Lock()
        self.gap = 1.0 / rps
        self.t = 0.0
        self.c.token                                    # 토큰은 한 번만 발급(분당 1회 제한)

    def _slot(self):
        """스레드 공용 유량 제한: 요청 시작 간격 ≥ 1/rps."""
        with self.lock:
            now = time.time()
            start = max(now, self.t + self.gap)
            self.t = start
        if start > now:
            time.sleep(start - now)

    def get(self, path, tr_id, params):
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
        for i in range(6):
            self._slot()
            try:
                r = self.local.s.get(f"{self.c.base}{path}", headers=self.c._headers(tr_id), params=params, timeout=15)
                body = r.json()
                if body.get("rt_cd") == "0":
                    return body
                if "초당" in (body.get("msg1") or ""):          # 유량 초과 → 잠시 쉼
                    time.sleep(1.0); continue
                return body
            except Exception as e:
                log("kis retry", tr_id, params.get("FID_INPUT_ISCD") or params.get("PDNO"), repr(e)[:100])
                time.sleep(2 * (i + 1))
        return {}


def fetch_index_kis():
    """index_close.parquet 를 한투 업종 일봉(FHKUP03500100)으로 최신일까지 이어 붙인다. KIS 코드 0001/1001/2001."""
    k = _KIS()
    f = DATA / "index_close.parquet"
    old = pd.read_parquet(f)
    a = old.index.max() + pd.Timedelta(days=1)
    z = pd.Timestamp(end_date())
    if a > z:
        return
    new = {}
    for name, kc in (("KOSPI", "0001"), ("KOSDAQ", "1001"), ("KOSPI200", "2001")):
        body = k.get("/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice", "FHKUP03500100", {
            "FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": kc, "FID_INPUT_DATE_1": a.strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": z.strftime("%Y%m%d"), "FID_PERIOD_DIV_CODE": "D"})
        rows = [r for r in (body.get("output2") or []) if r.get("stck_bsop_date")]
        new[name] = pd.Series({pd.Timestamp(r["stck_bsop_date"]): float(r["bstp_nmix_prpr"]) for r in rows})
    new = pd.DataFrame(new).dropna(how="all")
    if len(new):
        out = pd.concat([old, new[new.index >= a]]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out.to_parquet(f)
        log("index +", len(new), out.index.max().date())


def candidate_codes() -> list:
    """생존편향 제거용 후보: DART 에 종목코드가 있는 전 법인(상장폐지 포함) ∪ 2009~2010 pykrx 로 받은 종목."""
    codes = set(dart_corpcodes())
    for f in glob.glob(str(RAW / "*" / "*.parquet")):
        codes |= set(pd.read_parquet(f, columns=["code"])["code"])
    return sorted(c for c in codes if len(c) == 6)


def fetch_kis_info():
    k = _KIS()
    f = DATA / "kis_info.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame()
    done = set(have["code"]) if len(have) else set()
    keys = ["prdt_abrv_name", "mket_id_cd", "scty_grp_id_cd", "stck_kind_cd", "lstg_stqt", "scts_mket_lstg_dt",
            "kosdaq_mket_lstg_dt", "lstg_abol_dt", "idx_bztp_lcls_cd_name", "idx_bztp_mcls_cd_name",
            "idx_bztp_scls_cd_name", "std_idst_clsf_cd_name"]
    rows = []
    todo = [c for c in candidate_codes() if c not in done]
    for i, c in enumerate(todo):
        o = k.get("/uapi/domestic-stock/v1/quotations/search-stock-info", "CTPF1002R", {"PRDT_TYPE_CD": "300", "PDNO": c}).get("output") or {}
        rows.append({"code": c, **{kk: o.get(kk, "") for kk in keys}})
        if i % 500 == 0:
            log("kis info", i, len(todo))
    out = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
    out.to_parquet(f)
    log("kis info saved", len(out))


def kis_stocks() -> pd.DataFrame:
    """보통주(ST, 종류 101 또는 공란, 코드 끝자리 0)·코스피/코스닥·2009-06 이후 상장 기간이 있는 종목."""
    inf = pd.read_parquet(DATA / "kis_info.parquet")
    inf = inf[(inf.scty_grp_id_cd == "ST") & inf.mket_id_cd.isin(["STK", "KSQ"]) & inf.stck_kind_cd.isin(["101", ""])].copy()   # 종류코드 공란인 보통주가 많다
    ld = lambda s: pd.to_datetime(s.replace("", None), format="%Y%m%d", errors="coerce")
    inf["kospi_dt"], inf["kosdaq_dt"], inf["delist_dt"] = ld(inf.scts_mket_lstg_dt), ld(inf.kosdaq_mket_lstg_dt), ld(inf.lstg_abol_dt)
    inf = inf[inf.code.str[-1] == "0"]
    inf["list_dt"] = inf[["kospi_dt", "kosdaq_dt"]].min(axis=1)
    inf = inf[inf.delist_dt.isna() | (inf.delist_dt >= pd.Timestamp(CFG["data"]["start"]))]
    return inf.set_index("code")


def _kis_chart(k, c, a, z):
    rows, cur = [], a
    while cur <= z:
        nxt = min(cur + pd.Timedelta(days=139), z)
        body = k.get("/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice", "FHKST03010100", {
            "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": c, "FID_INPUT_DATE_1": cur.strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": nxt.strftime("%Y%m%d"), "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "0"})
        rows += [r for r in (body.get("output2") or []) if r.get("stck_bsop_date")]
        cur = nxt + pd.Timedelta(days=1)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    num = lambda s: pd.to_numeric(s, errors="coerce").values        # .values: RangeIndex 정렬로 NaN 되는 것 방지
    df = pd.DataFrame({
        "open": num(df.stck_oprc), "high": num(df.stck_hgpr), "low": num(df.stck_lwpr),
        "close": num(df.stck_clpr), "volume": num(df.acml_vol),
        "value": num(df.acml_tr_pbmn)}, index=pd.to_datetime(df.stck_bsop_date, format="%Y%m%d").values)
    df = df[~df.index.duplicated()].sort_index()
    return df[(df.index >= a) & (df.index <= z)]


def fetch_kis_prices(codes=None, workers=8):
    """종목별 수정주가 일봉 → data/kis_daily/{code}.parquet. 100영업일(≈140일) 창으로 끊어 받는다.
    이미 받은 종목은 마지막 10일을 겹쳐 이어 받고, 겹친 구간 종가가 다르면(새 분할·권리락으로 수정 기준 변경)
    전 구간을 다시 받는다. 스레드 여러 개 + 공용 유량 제한(15/s)."""
    from concurrent.futures import ThreadPoolExecutor
    k = _KIS()
    st = kis_stocks()
    codes = codes or list(st.index)
    d = DATA / "kis_daily"; d.mkdir(exist_ok=True)
    s0, e0 = pd.Timestamp(CFG["data"]["start"]), pd.Timestamp(end_date())

    def one(c):
        f = d / f"{c}.parquet"
        a = max(s0, st.at[c, "list_dt"] if pd.notna(st.at[c, "list_dt"]) else s0)
        z = min(e0, st.at[c, "delist_dt"] if pd.notna(st.at[c, "delist_dt"]) else e0)
        old = pd.read_parquet(f) if f.exists() else None
        if old is not None and len(old):
            if pd.notna(st.at[c, "delist_dt"]) or old.index.max() >= z:
                return "skip"
            new = _kis_chart(k, c, old.index.max() - pd.Timedelta(days=10), z)
            if new is None:
                return "none"
            # 기존 파일의 마지막 날은 18시 전 수집이면 미확정일 수 있어 비교에서 빼고 새 값으로 덮어쓴다
            ov = old["close"].iloc[:-1].reindex(new.index).dropna()
            if len(ov) and ((new["close"].reindex(ov.index) / ov - 1).abs() > 1e-3).any():
                df = _kis_chart(k, c, a, z); tag = "refetch"      # 수정 기준 변경
            else:
                df = pd.concat([old, new]); df = df[~df.index.duplicated(keep="last")].sort_index(); tag = "append"
        else:
            df = _kis_chart(k, c, a, z); tag = "full"
        if df is not None and len(df):
            df.to_parquet(f)
        return tag

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for tag in ex.map(one, codes):
            done += 1
            if done % 100 == 0:
                log("kis prices", done, "/", len(codes), tag)
    log("kis prices done", len(codes))


def build_panel_kis():
    """kis_daily → panel_*.parquet (build_panel 과 같은 필드).
    - 가격·거래량은 한투 수정주가(분할·권리락 반영). 거래대금은 원 단위 그대로.
    - 시총은 과거 상장주식수를 알 수 없어 '최종 상장주식수 × 수정종가' 로 근사한다
      (분할은 정확히 상쇄되지만 이후 유상증자·CB 전환분만큼 과거 시총을 과대평가). COVERAGE.md 명시.
    - 소속 시장: 코스닥 상장일 < 코스피 상장일 이면 코스피 상장일 전까지 코스닥(이전상장 반영)."""
    st = kis_stocks()
    days = trading_days()
    days = days[days >= pd.Timestamp(CFG["data"]["start"])]
    frames = {}
    for f in sorted((DATA / "kis_daily").glob("*.parquet")):
        if f.stem in st.index:
            frames[f.stem] = pd.read_parquet(f)
    fields = {}
    for k in ("open", "high", "low", "close", "volume", "value"):
        fields[k] = pd.DataFrame({c: v[k] for c, v in frames.items()}).reindex(days)
    close = fields["close"]
    # 상장 기간 안에서 행이 없는 날 = 거래정지(한투는 정지일을 건너뛴다) → 전일 종가 유지, 거래량 0
    first = close.apply(lambda s: s.first_valid_index()); last = close.apply(lambda s: s.last_valid_index())
    pos = np.arange(len(days))[:, None]
    fi = days.get_indexer(first.values); li = days.get_indexer(last.values)
    listed = pd.DataFrame((pos >= fi[None, :]) & (pos <= li[None, :]), index=days, columns=close.columns)
    halted = ((fields["volume"] == 0) | close.isna() | (fields["open"] == 0)) & listed
    adj_close = close.ffill().where(listed)
    shares = pd.to_numeric(st["lstg_stqt"], errors="coerce").reindex(close.columns)
    kospi = pd.DataFrame(0.0, index=days, columns=close.columns)
    for c in close.columns:
        r = st.loc[c]
        if r.mket_id_cd == "STK":
            if pd.notna(r.kosdaq_dt) and pd.notna(r.kospi_dt) and r.kosdaq_dt < r.kospi_dt:
                kospi.loc[days >= r.kospi_dt, c] = 1.0
            else:
                kospi[c] = 1.0
    out = {
        "adj_close": adj_close,
        "adj_open": fields["open"].where(~halted),
        "adj_high": fields["high"].where(~halted),
        "adj_low": fields["low"].where(~halted),
        "adj_volume": fields["volume"].where(~halted, 0.0).where(listed),
        "volume": fields["volume"].where(~halted, 0.0).where(listed),
        "value": fields["value"].where(~halted, 0.0).where(listed),
        "mktcap": adj_close * shares.values[None, :],
        "close": adj_close,
        "halted": halted.where(listed),
        "is_kospi": kospi.where(listed),
    }
    for k, v in out.items():
        v.to_parquet(DATA / f"panel_{k}.parquet")
    pd.DataFrame({"market": np.where(st["mket_id_cd"] == "STK", "KOSPI", "KOSDAQ")}, index=st.index) \
        .reindex(close.columns).to_parquet(DATA / "code_market.parquet")
    names = {c: st.at[c, "prdt_abrv_name"] for c in close.columns}
    json.dump(names, open(DATA / "names.json", "w", encoding="utf-8"), ensure_ascii=False)
    sec = st[["idx_bztp_mcls_cd_name", "idx_bztp_scls_cd_name", "std_idst_clsf_cd_name"]].reindex(close.columns)
    bad = lambda x: (not x) or x in ("제조", "제조업", "서비스업") or any(k in x for k in ("KOGI", "시가총액", "KRX", "코스닥", "우량", "벤처", "중견", "기술"))
    sector = [next((x for x in (m, s_) if not bad(x)), std or "") for m, s_, std in sec.itertuples(index=False)]
    pd.DataFrame({"sector": sector, "std_industry": sec["std_idst_clsf_cd_name"].values}, index=close.columns).to_parquet(DATA / "sector.parquet")
    log("kis panel", close.shape, int(listed.values.sum()), "listed stock-days")


# ───────────────────────── 패널 구성(수정주가) — pykrx 원시 경로(2009-06~2010-11 만 수집됨, 교차검증용) ─────────────────────────
def build_panel():
    """raw_daily → data/panel_{field}.parquet (행=일자, 열=종목). 수정 OHLC + 원종가·거래대금·시총·시장."""
    raw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(RAW / "*" / "*.parquet")))], ignore_index=True)
    raw = raw.drop_duplicates(["date", "code"], keep="last")
    wide = {c: raw.pivot(index="date", columns="code", values=c).sort_index()
            for c in ["open", "high", "low", "close", "volume", "value", "chg", "mktcap"]}
    close = wide["close"].astype(float)
    vol = wide["volume"].astype(float)
    halted = (vol == 0) | (wide["open"] == 0)               # 거래정지일: 시가 0, 거래량 0

    # 수정종가 지수 = Π(1+등락률). 거래정지일 등락률 0.
    # 등락률은 소수 둘째자리 반올림이라 누적하면 미세하게 표류한다 → 평소엔 원종가 비율, 권리락·분할처럼
    # 원종가 비율과 등락률이 0.5%p 넘게 어긋나는 날만 KRX 등락률을 쓴다.
    krx_chg = wide["chg"].astype(float) / 100.0
    raw_ret = close / close.ffill().shift(1) - 1
    chg = raw_ret.where((raw_ret - krx_chg).abs() <= 0.005, krx_chg).where(~halted, 0.0)
    listed = close.notna()
    growth = (1 + chg.fillna(0)).where(listed)
    adj_idx = growth.cumprod()
    # 마지막 관측 원종가에 맞춰 기준화 → 최근 가격은 실제 호가와 같다
    last_close = close.ffill().iloc[-1] if False else close.apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    last_idx = adj_idx.apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    adj_close = adj_idx * (last_close / last_idx)
    factor = (adj_close / close).where(~halted)
    factor = factor.ffill()

    out = {
        "adj_close": adj_close,
        "adj_open": (wide["open"].astype(float) * factor).where(~halted),
        "adj_high": (wide["high"].astype(float) * factor).where(~halted),
        "adj_low": (wide["low"].astype(float) * factor).where(~halted),
        "volume": vol.where(~halted, 0.0).where(listed),
        "adj_volume": (vol / factor).where(~halted, 0.0).where(listed),   # 분할 후 거래량과 비교 가능하게
        "value": wide["value"].astype(float).where(listed),
        "mktcap": wide["mktcap"].astype(float),
        "close": close,
        "halted": halted.where(listed),
    }
    # 일자별 소속 시장(코스닥→코스피 이전상장 반영)
    out["is_kospi"] = (raw.assign(k=(raw["market"] == "KOSPI").astype(float))
                       .pivot(index="date", columns="code", values="k").sort_index())
    for k, v in out.items():
        v.to_parquet(DATA / f"panel_{k}.parquet")
    mk = raw.sort_values("date").groupby("code")["market"].last()
    mk.to_frame().to_parquet(DATA / "code_market.parquet")
    log("panel", close.shape, close.index.min().date(), close.index.max().date())


def load_panel(fields=("adj_open", "adj_high", "adj_low", "adj_close", "adj_volume", "value", "mktcap", "halted", "is_kospi")):
    return {f: pd.read_parquet(DATA / f"panel_{f}.parquet") for f in fields}


def validate_adjustment(codes=("005930", "035420", "000660", "005380", "034220")):
    """등락률 누적 수정주가 vs pykrx 종목별 수정주가(2014-07~) 비교. 최대 괴리를 돌려준다."""
    stock = _krx()
    adj = pd.read_parquet(DATA / "panel_adj_close.parquet")
    rows = []
    for c in codes:
        ref = _retry(stock.get_market_ohlcv, "20100101", end_date().replace("-", ""), c, adjusted=True)["종가"]
        ref = ref[ref > 0]
        mine = adj[c].reindex(ref.index)
        rel = (mine / ref - 1).abs()
        rows.append({"code": c, "n": int(rel.notna().sum()), "max_abs_diff": float(rel.max()),
                     "median_abs_diff": float(rel.median()), "from": str(ref.index.min().date())})
    return pd.DataFrame(rows)


def universe_mask(px: dict, cfg: dict, names: dict, markets=None) -> pd.DataFrame:
    """일자×종목 유니버스(t일 종가 기준 정보만 사용).
    - 소속 시장(markets), 보통주만(코드 끝자리 0), 스팩·리츠·인프라펀드 등 이름 패턴 제외
    - 20일 평균 거래대금 ≥ 기준, 시총 ≥ 기준, 최근 N일 거래정지 없음
    관리종목·투자경고 이력은 pykrx 로 과거 시점을 얻을 수 없어 적용하지 못한다(COVERAGE.md 명시)."""
    u = cfg["universe"]
    markets = markets or u["markets"]
    close = px["adj_close"]
    kospi = px["is_kospi"].reindex_like(close)
    if set(markets) == {"KOSPI"}:
        mk = kospi == 1
    elif set(markets) == {"KOSDAQ"}:
        mk = kospi == 0
    else:
        mk = kospi.notna()
    pats = u["exclude_name_patterns"]
    ok_code = pd.Series({c: (c[-1] == "0") and not any(p in (names.get(c) or "") for p in pats) for c in close.columns})
    val20 = px["value"].rolling(20, min_periods=15).mean()
    halt = px["halted"].astype(float).rolling(u["halt_lookback"], min_periods=1).max() > 0
    m = (mk & close.notna() & (val20 >= u["min_value_20d_krw"]) & (px["mktcap"] >= u["min_mktcap_krw"]) & ~halt)
    return m & ok_code.reindex(close.columns).fillna(False).values[None, :]


# ───────────────────────── 종목명 ─────────────────────────
def dart_corpcodes() -> dict:
    f = DATA / "dart_corp.json"
    if not f.exists():
        r = requests.get("https://opendart.fss.or.kr/api/corpCode.xml",
                         params={"crtfc_key": os.environ["DART_API_KEY"]}, timeout=60)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        root = ET.fromstring(z.read(z.namelist()[0]))
        m = {}
        for e in root.iter("list"):
            sc = (e.findtext("stock_code") or "").strip()
            if sc:
                m[sc] = {"corp_code": e.findtext("corp_code"), "name": e.findtext("corp_name")}
        json.dump(m, open(f, "w", encoding="utf-8"), ensure_ascii=False)
    return json.load(open(f, encoding="utf-8"))


def fetch_names():
    """종목명: pykrx(현재·과거 상장) 우선, 없으면 DART 법인명."""
    stock = _krx()
    codes = pd.read_parquet(DATA / "code_market.parquet").index
    dart = dart_corpcodes()
    f = DATA / "names.json"
    names = json.load(open(f, encoding="utf-8")) if f.exists() else {}
    for i, c in enumerate(codes):
        if c in names:
            continue
        try:
            n = stock.get_market_ticker_name(c)
            n = n if isinstance(n, str) and n else None
        except Exception:
            n = None
        names[c] = n or dart.get(c, {}).get("name") or ""
        if i % 200 == 0:
            json.dump(names, open(f, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(names, open(f, "w", encoding="utf-8"), ensure_ascii=False)
    log("names", len(names), "empty", sum(1 for v in names.values() if not v))


def load_names() -> dict:
    return json.load(open(DATA / "names.json", encoding="utf-8"))


# ───────────────────────── 업종 ─────────────────────────
def fetch_sector(dates=("20120102", "20160104", "20200102", "20230102", None)):
    """KRX 업종지수 구성종목 역산. 대형/중형/소형(1002~1004, 2002~2004)·umbrella 는 제외.
    여러 시점을 돌려 상장폐지 종목도 업종을 얻는다(최근 시점이 우선)."""
    stock = _krx()
    days = trading_days()
    rows = []
    for d in dates:
        d = d or days.max().strftime("%Y%m%d")
        d = days[days >= pd.Timestamp(d)][0].strftime("%Y%m%d")
        for mkt in ("KOSPI", "KOSDAQ"):
            for idx in _retry(stock.get_index_ticker_list, d, market=mkt):
                n = int(idx)
                if mkt == "KOSPI" and not (1005 <= n <= 1027):
                    continue
                if mkt == "KOSDAQ" and not (2012 <= n <= 2099):   # 2001 전체, 2002~2004 규모, 2007~ 벤처/기술 등은 이름으로 거른다
                    continue
                name = stock.get_index_ticker_name(idx)
                if any(x in name for x in ("대형", "중형", "소형", "우량", "벤처", "기술", "150", "글로벌", "프리미어", "코스닥")):
                    continue
                try:
                    mem = _retry(stock.get_index_portfolio_deposit_file, idx, d)
                except Exception:
                    continue
                for c in mem:
                    rows.append({"asof": d, "market": mkt, "index": idx, "sector": name, "code": c})
        log("sector", d, len(rows))
    df = pd.DataFrame(rows)
    df.to_parquet(DATA / "sector_raw.parquet")
    # 우선순위: 최근 시점 > 좁은 업종(umbrella '제조' 는 마지막)
    df["umb"] = df["sector"].str.contains("제조").astype(int)
    df = df.sort_values(["asof", "umb"], ascending=[False, True])
    sec = df.drop_duplicates("code")[["code", "sector", "market"]].set_index("code")
    sec.to_parquet(DATA / "sector.parquet")
    log("sector map", len(sec))


def cyclical_map(cfg=None) -> pd.Series:
    """경기민감 여부: 업종명 + 표준산업분류를 합쳐 config 키워드와 부분일치."""
    cfg = cfg or CFG
    sec = pd.read_parquet(DATA / "sector.parquet")
    txt = sec["sector"].fillna("") + "|" + sec["std_industry"].fillna("")
    exc = cfg.get("cyclical_exclude", [])
    return txt.apply(lambda x: any(k in x for k in cfg["cyclical_sectors"]) and not any(k in x for k in exc))


def load_sector() -> pd.Series:
    sec = pd.read_parquet(DATA / "sector.parquet")["sector"]
    codes = pd.read_parquet(DATA / "code_market.parquet").index
    s = sec.reindex(codes)
    miss = s.isna()
    s[miss] = [sec.get(c[:5] + "0") for c in s.index[miss]]    # 우선주 → 보통주 업종 상속
    return s


# ───────────────────────── DART 분기 재무 ─────────────────────────
REPRT = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}
ACCTS = {"매출액": "sales", "영업이익": "op", "당기순이익(손실)": "ni", "당기순이익": "ni"}


def _amt(x):
    x = (x or "").replace(",", "").strip()
    if x in ("", "-"):
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def fetch_dart(years=None):
    """fnlttMultiAcnt(다중회사 주요계정)로 매출·영업이익·당기순이익 수집. 2015 사업보고서 / 2016Q1~.
    rcept_no 앞 8자리 = 접수일 → 포인트인타임 날짜. 연도·보고서별 json 캐시."""
    key = os.environ["DART_API_KEY"]
    corp = dart_corpcodes()
    cc = sorted({v["corp_code"] for v in corp.values()})   # DART 에 종목코드가 있는 전 법인(상장폐지 포함)
    cache = DATA / "dart_raw"; cache.mkdir(exist_ok=True)
    years = years or range(2015, int(end_date()[:4]) + 1)
    for y in years:
        for rc in REPRT:
            f = cache / f"{y}_{rc}.json"
            if f.exists() and y < int(end_date()[:4]) - 1:
                continue
            got = []
            for i in range(0, len(cc), 100):
                for t in range(4):
                    try:
                        r = requests.get("https://opendart.fss.or.kr/api/fnlttMultiAcnt.json", timeout=60, params={
                            "crtfc_key": key, "corp_code": ",".join(cc[i:i + 100]), "bsns_year": y, "reprt_code": rc}).json()
                        break
                    except Exception as e:
                        log("dart retry", y, rc, e); time.sleep(5)
                if r.get("status") == "020":
                    raise SystemExit("DART 일일 한도 초과")
                got += [x for x in r.get("list", []) if x.get("sj_div") == "IS" and x.get("account_nm") in ACCTS]
                time.sleep(0.15)
            json.dump(got, open(f, "w", encoding="utf-8"), ensure_ascii=False)
            log("dart", y, rc, len(got))


def build_fundamentals():
    """dart_raw → data/fin_quarterly.parquet : code, fy, q, fs_div, sales/op/ni(단일분기, 십억원), avail_date.
    - 분기(1~3Q) 보고서 thstrm_amount = 해당 3개월. frmtrm_amount = 전년 동기 3개월.
    - 4Q = 사업보고서 연간 − (1Q+2Q+3Q). 전년 4Q 도 전년 연간(frmtrm) − 전년 1~3Q 로 만든다.
    - 연결(CFS) 우선, 없으면 별도(OFS).
    - 당기순이익은 '지배주주' 가 아닌 연결 당기순이익(다중회사 주요계정 API 한계)."""
    rows = []
    for f in sorted((DATA / "dart_raw").glob("*.json")):
        y, rc = f.stem.split("_")
        for x in json.load(open(f, encoding="utf-8")):
            rows.append({"code": x["stock_code"], "fy": int(y), "q": REPRT[rc], "fs": x["fs_div"],
                         "acct": ACCTS[x["account_nm"]], "cur": _amt(x.get("thstrm_amount")),
                         "prev": _amt(x.get("frmtrm_amount")),
                         "rcept": pd.Timestamp(x["rcept_no"][:8])})
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["code", "fy", "q", "fs", "acct"], keep="last")
    # 연결 우선
    has_cfs = df[df.fs == "CFS"].groupby(["code", "fy", "q"]).size()
    key = df.set_index(["code", "fy", "q"]).index
    df = df[(df.fs == "CFS") | ~key.isin(has_cfs.index)]
    cur = df.pivot_table(index=["code", "fy", "q"], columns="acct", values="cur", aggfunc="last")
    prev = df.pivot_table(index=["code", "fy", "q"], columns="acct", values="prev", aggfunc="last")
    rcept = df.groupby(["code", "fy", "q"])["rcept"].max()

    # 단일분기 값 테이블: (code, fy, q) → 당기값. 전년 동기는 prev 로 (fy-1, q) 를 채운다(당기 보고서에 없는 과거 분기 보완)
    val = cur.copy()
    back = prev.copy()
    back.index = pd.MultiIndex.from_tuples([(c, fy - 1, q) for c, fy, q in back.index], names=val.index.names)
    val = val.combine_first(back)
    # 단위 오기재(원 대신 천원·백만원 등으로 ×1e3~1e6 튄 값) 제거: 종목 자신의 중앙값 대비 300배 넘으면 NaN.
    # Supabase fin_operating_profit 와 대조해 찾은 사례(060310 2022, 007720 2024 등).
    med = val.abs().groupby(level="code").transform("median")
    bad = (val.abs() > med * 300) & (val.abs() > 1e11)
    log("unit outliers removed", int(bad.values.sum()))
    val = val.mask(bad)
    # 4Q = 연간 − 1~3Q  (q=4 행은 지금 '연간' 값)
    out = val.copy()
    for (c, fy, q) in val.index:
        if q != 4:
            continue
        try:
            s3 = sum(val.loc[(c, fy, k)] for k in (1, 2, 3))
            out.loc[(c, fy, 4)] = val.loc[(c, fy, 4)] - s3
        except KeyError:
            out.loc[(c, fy, 4)] = np.nan
    out = out / 1e9        # 원 → 십억원
    out["avail"] = rcept.reindex(out.index)    # 전년 동기(back) 로만 채운 행은 avail 없음 → 당해 보고서 시점에만 쓰인다
    out = out.reset_index()
    out.to_parquet(DATA / "fin_quarterly.parquet")
    log("fundamentals", len(out), out["code"].nunique(), "codes", out.fy.min(), out.fy.max())


# ───────────────────────── 투자자 수급(신호 종목만) ─────────────────────────
def fetch_flows(codes):
    """외국인+기관 일별 순매수대금(원). 종목당 ~20초라 신호가 난 종목만 받는다. 종목별 parquet 캐시."""
    stock = _krx()
    d = DATA / "flows"; d.mkdir(exist_ok=True)
    s, e = CFG["data"]["start"].replace("-", ""), end_date().replace("-", "")
    for i, c in enumerate(codes):
        f = d / f"{c}.parquet"
        if f.exists():
            continue
        try:
            v = _retry(stock.get_market_trading_value_by_date, s, e, c)
            v[["기관합계", "외국인합계"]].rename(columns={"기관합계": "inst", "외국인합계": "frgn"}).to_parquet(f)
        except Exception as ex:
            log("flow fail", c, ex)
        if i % 20 == 0:
            log("flows", i, len(codes))


def kis_recent_flows(codes) -> pd.DataFrame:
    """한투 종목별 투자자매매동향(일별) FHPTJ04160001: 최근 30영업일 외국인+기관 순매수대금(원). 스크리너 전용.
    (과거 이력은 pykrx 가 유일했는데 KRX 차단으로 백테스트 수급 점수는 비활성)"""
    k = _KIS()
    today = end_date().replace("-", "")
    cols = {}
    for c in codes:
        body = k.get("/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily", "FHPTJ04160001", {
            "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": c, "FID_INPUT_DATE_1": today,
            "FID_ORG_ADJ_PRC": "0", "FID_ETC_CLS_CODE": ""})
        rows = body.get("output2") or []
        s = {}
        for r in rows:
            try:
                s[pd.Timestamp(r["stck_bsop_date"])] = (float(r.get("frgn_ntby_tr_pbmn") or 0) + float(r.get("orgn_ntby_tr_pbmn") or 0)) * 1e6
            except (KeyError, ValueError):
                pass
        if s:
            cols[c] = pd.Series(s)
    return pd.DataFrame(cols).sort_index()


def load_flows(codes) -> pd.DataFrame:
    """외국인+기관 합산 순매수대금 wide (없는 종목은 열 없음)."""
    d = DATA / "flows"
    cols = {}
    for c in codes:
        f = d / f"{c}.parquet"
        if f.exists():
            v = pd.read_parquet(f)
            cols[c] = v["inst"] + v["frgn"]
    return pd.DataFrame(cols)


# ───────────────────────── 테마(Theme Master Core) ─────────────────────────
def load_themes() -> pd.DataFrame:
    fs = sorted(glob.glob(str((ROOT / CFG["paths"]["theme_master_glob"]).resolve())))
    if not fs:
        return pd.DataFrame(columns=["code", "theme"])
    t = pd.read_csv(fs[-1], dtype=str)
    return t


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch-index":
        fetch_index()
    elif cmd == "fetch-prices":
        fetch_prices(sys.argv[2:] or None)
    elif cmd == "build":
        build_panel()
    elif cmd == "validate":
        print(validate_adjustment().to_string())
    elif cmd == "fetch-names":
        fetch_names()
    elif cmd == "fetch-sector":
        fetch_sector()
    elif cmd == "fetch-dart":
        fetch_dart()
    elif cmd == "build-fund":
        build_fundamentals()
    elif cmd == "kis-info":
        fetch_kis_info()
    elif cmd == "kis-prices":
        fetch_kis_prices(sys.argv[2:] or None)
    elif cmd == "kis-build":
        build_panel_kis()
    else:
        print(__doc__)
