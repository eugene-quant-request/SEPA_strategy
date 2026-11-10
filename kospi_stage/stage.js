/* 미너비니 4단계(Stage Analysis) 판정 엔진 — 『초수익 성장주 투자』 5장 기준.
 * 브라우저와 node 양쪽에서 쓴다(node: require('./stage.js')).
 * 입력: {d:[날짜], o,h,l,c,v:[숫자]}  출력: 날짜별 단계·조건 충족 여부.
 * 모든 값은 t일 종가까지의 정보만 쓴다(미래 정보 없음).
 */
(function (root) {
  const P = {
    rise: 0.005,       // 200일선 21영업일 변화율이 +0.5% 넘으면 상승, -0.5% 밑이면 하락, 사이는 횡보
    nearLow: 0.15,     // 52주 최저가 대비 +15% 이내면 "신저가 근처"
    nearMa: 0.07,      // 200일선 ±7% 이내면 "200일선 근처"
    volExp: 1.3,       // 20일 변동성이 1년 중앙값의 1.3배 넘으면 "변동성 확대"
    weeks: 13,         // 주봉 거래량 비교 구간(약 한 분기)
    avgWeeks: 50,      // "평균 이상 거래량"의 기준 평균(직전 50주)
    swing: 63,         // 고점·저점 비교 구간(최근 63일 vs 그 앞 63일)
    confirm: 10,       // 같은 후보 단계가 10영업일 이어져야 단계 전환
  };

  function sma(a, n) {
    const out = new Array(a.length).fill(null); let s = 0;
    for (let i = 0; i < a.length; i++) {
      s += a[i]; if (i >= n) s -= a[i - n];
      if (i >= n - 1) out[i] = s / n;
    }
    return out;
  }
  function rmax(a, i0, i1) { let m = -Infinity; for (let i = i0; i <= i1; i++) if (a[i] > m) m = a[i]; return m; }
  function rmin(a, i0, i1) { let m = Infinity; for (let i = i0; i <= i1; i++) if (a[i] < m) m = a[i]; return m; }
  function median(arr) { const s = arr.filter(x => x != null).sort((x, y) => x - y); return s.length ? s[s.length >> 1] : null; }
  function weekKey(ds) {           // 해당 주 월요일 (UTC 기준 문자열)
    const t = new Date(ds + 'T00:00:00Z'); const wd = (t.getUTCDay() + 6) % 7;
    return new Date(t.getTime() - wd * 864e5).toISOString().slice(0, 10);
  }

  /* 지표 계산 */
  function indicators(D) {
    const n = D.c.length, c = D.c;
    const ma50 = sma(c, 50), ma150 = sma(c, 150), ma200 = sma(c, 200);
    const vol50 = sma(D.v, 50), vol200 = sma(D.v, 200);
    const slope = ma200.map((m, i) => (i >= 21 && m != null && ma200[i - 21] != null) ? m / ma200[i - 21] - 1 : null);
    // 20일 실현변동성
    const ret = c.map((x, i) => i ? Math.log(x / c[i - 1]) : 0);
    const rv = new Array(n).fill(null);
    for (let i = 20; i < n; i++) { let s = 0, s2 = 0; for (let k = i - 19; k <= i; k++) { s += ret[k]; s2 += ret[k] * ret[k]; } rv[i] = Math.sqrt(Math.max(0, s2 / 20 - (s / 20) ** 2)) * Math.sqrt(252); }
    // 주봉: 주가 끝난 날(다음 거래일이 다른 주) 인덱스에 확정
    const wk = D.d.map(weekKey), W = [];        // {end:i, close, vol}
    const lastWeekBefore = new Array(n).fill(-1);  // i일 기준 확정된 마지막 주 번호
    let vs = 0;
    for (let i = 0; i < n; i++) {
      vs += D.v[i];
      const closes = i === n - 1 ? false : wk[i + 1] !== wk[i];
      if (closes) { W.push({ end: i, close: c[i], vol: vs }); vs = 0; }
      lastWeekBefore[i] = W.length - 1;
    }
    return { ma50, ma150, ma200, vol50, vol200, slope, rv, W, lastWeekBefore };
  }

  /* i일의 조건 판정 */
  function conditions(D, I, i) {
    const c = D.c[i], m50 = I.ma50[i], m150 = I.ma150[i], m200 = I.ma200[i], sl = I.slope[i];
    if (i < 260 || m200 == null || sl == null) return null;
    // 고점·저점 계단
    const S = P.swing;
    const hA = rmax(D.h, i - S + 1, i), hB = rmax(D.h, i - 2 * S + 1, i - S);
    const lA = rmin(D.l, i - S + 1, i), lB = rmin(D.l, i - 2 * S + 1, i - S);
    // 주봉 거래량
    const wi = I.lastWeekBefore[i], W = I.W;
    let upV = 0, upN = 0, dnV = 0, dnN = 0, upHeavy = 0, dnHeavy = 0, dnHeavy8 = 0;
    if (wi >= P.weeks + P.avgWeeks) {
      for (let k = wi - P.weeks + 1; k <= wi; k++) {
        let avg = 0; for (let j = k - P.avgWeeks; j < k; j++) avg += W[j].vol; avg /= P.avgWeeks;
        const up = W[k].close > W[k - 1].close, heavy = W[k].vol > avg;
        if (up) { upV += W[k].vol; upN++; if (heavy) upHeavy++; }
        else { dnV += W[k].vol; dnN++; if (heavy) { dnHeavy++; if (k > wi - 8) dnHeavy8++; } }
      }
    }
    const upAvg = upN ? upV / upN : 0, dnAvg = dnN ? dnV / dnN : 0;
    const low52 = rmin(D.l, i - 249, i), high52 = rmax(D.h, i - 249, i);
    const lowPrev = rmin(D.l, i - 249, i - 42);              // 최근 42일 이전까지의 52주 저점
    const rvMed = median(I.rv.slice(i - 249, i + 1));
    const sl63 = I.slope[i - 63];
    const x = {
      // 2단계(상승) 체크리스트 — 책의 1→2단계 전환 요건 6개 + 단기선 정배열
      c1: c > m150 && c > m200,
      c2: m150 > m200,
      c3: sl > P.rise,
      c4: hA > hB && lA > lB,
      c5: upAvg > dnAvg,
      c6: upHeavy > dnHeavy,
      c7: m50 > m150,
      // 4단계(하락) 체크리스트
      d1: c < m200,
      d2: sl < -P.rise,
      d3: c <= low52 * (1 + P.nearLow),
      d4: hA < hB && lA < lB,
      d5: dnAvg > upAvg,
      d6: dnHeavy > upHeavy,
      d7: m50 < m150,
      // 3단계(고점) 경고 신호
      e1: rvMed != null && I.rv[i] > rvMed * P.volExp,
      e2: c < m50,
      e3: sl < P.rise || (sl63 != null && sl63 > 0 && sl < sl63 * 0.5),
      e4: dnHeavy8 >= 2,
      // 1단계(바닥) 신호
      f1: Math.abs(sl) <= P.rise,
      f2: Math.abs(c / m200 - 1) <= P.nearMa,
      f3: I.vol50[i] != null && I.vol200[i] != null && I.vol50[i] < I.vol200[i],
      f4: low52 >= lowPrev,                                   // 최근 두 달 신저가 없음
    };
    x.s2 = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'].filter(k => x[k]).length;
    x.s4 = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7'].filter(k => x[k]).length;
    x.s3 = ['e1', 'e2', 'e3', 'e4'].filter(k => x[k]).length;
    x.s1 = ['f1', 'f2', 'f3', 'f4'].filter(k => x[k]).length;
    x.core2 = x.c1 && x.c2 && x.c3;
    x.core4 = x.d1 && x.d2;
    Object.assign(x, { close: c, ma50: m50, ma150: m150, ma200: m200, slope: sl, low52, high52, rv: I.rv[i], rvMed, upAvg, dnAvg, upHeavy, dnHeavy });
    return x;
  }

  /* 직전 확정 단계 prev 와 오늘 조건 x 로 후보 단계 */
  function candidate(prev, x) {
    const strong2 = x.core2 && x.s2 >= 6, strong4 = x.core4 && x.s4 >= 5;
    switch (prev) {
      case 2: if (x.core2) return x.s3 >= 3 ? 3 : 2; return strong4 ? 4 : 3;
      case 3: if (strong2 && x.s3 <= 1) return 2; if (strong4) return 4; return 3;
      case 4: if (x.core4 && x.s1 < 3) return 4; return strong2 ? 2 : 1;
      default: if (strong2) return 2; if (strong4) return 4; return 1;
    }
  }

  function classify(D) {
    const I = indicators(D), n = D.c.length;
    const stage = new Array(n).fill(null), cand = new Array(n).fill(null), X = new Array(n).fill(null);
    let cur = null, pend = null, cnt = 0;
    for (let i = 0; i < n; i++) {
      const x = conditions(D, I, i); X[i] = x;
      if (!x) continue;
      const k = candidate(cur, x); cand[i] = k;
      if (cur == null) { cur = k; }
      else if (k !== cur) {
        if (k === pend) cnt++; else { pend = k; cnt = 1; }
        if (cnt >= P.confirm) { cur = k; pend = null; cnt = 0; }
      } else { pend = null; cnt = 0; }
      stage[i] = cur;
    }
    return { I, X, stage, cand };
  }

  /* 구간 묶기 */
  function segments(D, stage) {
    const seg = [];
    for (let i = 0; i < stage.length; i++) {
      if (stage[i] == null) continue;
      const s = seg[seg.length - 1];
      if (s && s.stage === stage[i] && s.end === i - 1) s.end = i;
      else seg.push({ stage: stage[i], start: i, end: i });
    }
    seg.forEach(s => { s.from = D.d[s.start]; s.to = D.d[s.end]; s.days = s.end - s.start + 1; s.ret = D.c[s.end] / D.c[s.start] - 1; });
    return seg;
  }

  /* 미래 경로 붙이기: days 영업일 동안 총 ret 만큼 로그선형으로 이동, 거래량은 최근 50일 평균 × vmult */
  function extend(D, days, ret, vmult) {
    const n = D.c.length, last = D.c[n - 1];
    const vAvg = D.v.slice(n - 50).reduce((a, b) => a + b, 0) / 50;
    const E = { d: D.d.slice(), o: D.o.slice(), h: D.h.slice(), l: D.l.slice(), c: D.c.slice(), v: D.v.slice() };
    let t = new Date(D.d[n - 1] + 'T00:00:00Z');
    for (let k = 1; k <= days; k++) {
      do { t = new Date(t.getTime() + 864e5); } while (t.getUTCDay() === 0 || t.getUTCDay() === 6);
      const p = last * Math.pow(1 + ret, k / days), pp = E.c[E.c.length - 1];
      E.d.push(t.toISOString().slice(0, 10)); E.o.push(pp); E.c.push(p);
      E.h.push(Math.max(p, pp) * 1.006); E.l.push(Math.min(p, pp) * 0.994); E.v.push(vAvg * vmult);
    }
    return E;
  }

  const api = { P, sma, indicators, conditions, candidate, classify, segments, extend };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Stage = api;
})(typeof window !== 'undefined' ? window : globalThis);
