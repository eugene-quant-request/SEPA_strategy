# 초수익 성장주 투자(SEPA) — 코스피 도입

마크 미너비니의 SEPA(추세 템플릿 · 펀더멘털 · 촉매 · VCP 진입 · 청산 규칙)를 코스피에 규칙화하고
일일 스크리너, 포인트인타임 백테스트, 결과 리포트를 만든다. 작업 지시서는 [PROMPT.md](PROMPT.md).

## 결론 요약

> 질문: 미너비니 방식이 코스피에서 초과수익을 내는가, 낸다면 어떤 조정이 필요한가?

- **책 그대로는 안 된다.** 추세 + 펀더멘털 + VCP + 국면 필터를 다 겹치면 2016-06 이후 10년간 진입이 24건뿐이다.
  평균 보유 종목은 0.09개로 사실상 현금이었다.
- **추세 템플릿 단독에는 엣지가 없다.** 돌파 신호 60일 초과수익은 표본 내 +0.1%, 표본 외 −0.15%.
  2011년 이후 성과는 연 −0.6%, MDD −73%.
- **펀더멘털 필터는 확실히 효과가 있다.** 같은 돌파 신호의 60일 초과수익이 통과 종목 +3.5%, 미통과 종목 −1.5%.
  표본 외에서도 유지됐고, 효과는 대부분 경기민감주에서 나왔다.
- **VCP는 신호의 질은 좋지만 너무 드물다.** 60일 초과수익은 표본 내 +7.3%, 표본 외 +3.3%이지만 신호가 연 18건뿐이다.
  2023년 이후 주도주 10종목에서는 VCP 신호가 1건만 나왔다.
- **손절은 8%가 맞다.** 5%면 연 −2.2%, 7%면 6.0%, 8%면 10.1%였다(실험 B). 표본 내 최적값이었고 표본 외에서도 유지됐다.
- 가장 나은 조합은 **추세 템플릿 + 펀더멘털 + 50일 신고가 돌파 + 손절 8%**로 연 10.1%, 샤프 0.49다.
  같은 기간 KOSPI(13.1%)와 KOSPI200(16.0%)에는 못 미쳤다(2025~26 대형주 급등). 동일가중 코스피(1.8%)는 크게 앞선다.

질문 7개에 대한 전체 답은 [output/conclusions.md](output/conclusions.md), 표와 차트는 `output/backtest_report.html`에 있다.

## 실행

```bash
python data_loader.py kis-info      # 종목 기본정보(시장·상장/폐지일·업종·주식수), ~20분
python data_loader.py kis-prices    # 종목별 수정주가 일봉 2009-06~ (상장폐지 포함), ~6시간
python data_loader.py kis-build     # 패널(parquet) 구성
python data_loader.py fetch-dart    # DART 분기 재무(FY2015, 2016Q1~), ~65분
python data_loader.py build-fund    # 분기 재무 정리(4Q 역산, 단위 오기재 제거)
python run_backtest.py --rebuild    # 실험 A~H → output/experiments.pkl
python report.py                    # backtest_report.html, trades.csv, top_trades.csv, vcp_samples/, 티어시트
python run_daily.py --update        # 최신일까지 이어 받고 output/screen_YYYYMMDD.csv
python run_daily.py --kosdaq        # 코스닥 포함 스크린
```

필요한 환경변수: `hts_api` / `hts_api_secret`(한투 실전 키, `../kis_api/kis_client.py` 사용), `DART_API_KEY`.
모든 임계값은 [config.yaml](config.yaml)에 있다.

## 구성

| 파일 | 역할 |
|---|---|
| `config.yaml` | 유니버스·추세·펀더멘털·VCP·국면·백테스트 파라미터, 연도별 증권거래세 |
| `data_loader.py` | 한투·DART 수집, 패널 구성, 유니버스 마스크, 업종·경기민감 판정 |
| `trend.py` | 추세 템플릿 8조건, 유니버스 내 RS 백분위, 4단계 판정(규칙은 코드 주석) |
| `fundamentals.py` | 포인트인타임 YoY 성장·가속·코드 33·감속 경고 |
| `catalyst.py` | 60일 신고가·수급·테마 점수(우선순위용, 필터 아님) |
| `vcp.py` | 수축 분해(H→L 재귀)·VCP·파워 플레이·단순 돌파 탐지 |
| `regime.py` | KOSPI/동일가중 200일선 국면, 브레드스, 유진 리서치 디텍터 |
| `backtest.py` | 일별 이벤트 시뮬레이션(t+1 시가 체결, 갭 하락은 시가 손절, 2R/3R 손절 상향, 50일선 대량 이탈 청산) |
| `run_backtest.py` | 실험 A~H와 변형 |
| `report.py` | HTML 리포트, 이벤트 스터디, VCP 샘플 차트, tossquant 티어시트 |
| `run_daily.py` | 일일 스크리너 |

## 데이터

자세한 커버리지·검증·한계는 [data/COVERAGE.md](data/COVERAGE.md)에 있다. 요점:

- **가격 소스는 한투 Open API다.** 처음에는 pykrx로 받았는데, KRX가 자동화 대량 조회로 판단해 IP를 1일 제한했다.
  재시도는 멈췄다. 한투 일봉은 수정주가이고 2009년까지 거슬러 가며, 상장폐지 종목도 내려준다(보통주 3,405개, 그중 폐지 776개).
- **수정주가는 교차검증했다.** KRX 등락률을 누적한 값과 한투 수정종가의 비율이 종목별로 상수다(변동계수 중앙값 0.0015%).
  삼성전자 50:1 분할 구간도 연속이다.
- **재무는 DART 다중회사 주요계정**에서 받았다. 매출·영업이익·연결 당기순이익이고, 접수일 다음 영업일부터 쓴다.
  Supabase와 대조하면 영업이익이 1% 이내로 일치하는 비율이 89%다.
- **펀더멘털이 들어간 실험은 2016-06부터** 돌렸다. DART API의 분기 재무가 2016Q1부터라서다.
- **비활성화한 요소**: 과거 외국인·기관 수급, 잠정실적 서프라이즈·컨센서스. 추정값으로 채우지 않았다.
- **근사한 요소**: 과거 시총은 최종 상장주식수 × 수정종가로 계산했다. 당기순이익은 지배주주 순이익이 아니다.

## 구현 중 결정

- **흑자전환**(전년 동기 적자 → 흑자)은 EPS 성장률 판정에서 +∞로 본다. 표시값은 NaN으로 둔다.
  끄면 D의 진입이 24건에서 9건으로 준다(`fundamentals.turnaround_as_growth`).
- **VCP 피벗 근접 조건**(피벗 ≥ 베이스 고점 × 0.90)은 시험만 하고 쓰지 않는다.
  육안으로는 책의 그림에 가깝지만, 표본 내 초과수익을 낮췄다(`vcp.pivot_near_high: 0`).
- **200일선 아래 종가면 다음날 매도**한다. 책의 "4단계 종목은 보유하지 않는다"를 반영한 규칙이다.
- **손절폭 확장 실험**: 12%와 15%는 책의 상한(10%)을 넘는다.

## 산출물

- `output/backtest_report.html` — 결론, 실험표, 이벤트 스터디, 국면, 주도주 타임라인, 상위 거래 20, VCP 샘플, 데이터 한계
- `output/tearsheet_D.html`, `output/tearsheet_A_full.html`, `output/tearsheet_B8.html` — tossquant 티어시트(일별 수익률, periods_per_year=252)
- `output/trades.csv` — 전 실험 거래 로그(진입·청산 사유, R 배수, 진입 시점 펀더멘털·VCP 정보)
- `output/top_trades.csv`, `output/leaders_timeline.csv`
- `output/vcp_samples/*.png` — VCP 탐지 샘플 20장
- `output/screen_YYYYMMDD.csv` — 일일 후보
