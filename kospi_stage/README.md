# 코스피 4단계 국면 판정 (미너비니 Stage Analysis)

『초수익 성장주 투자』 5장의 1~4단계 특징을 코스피 일봉 조건으로 옮겨 날짜별 단계를 판정하고,
앞으로 지수가 움직이는 경로(기간·등락률·거래량)를 넣으면 몇 단계가 되는지 다시 판정해 보여 준다.

## 갱신
```
python fetch_kospi.py   # 한투 FHKUP03500100 코스피 일봉, 최근 7일 덮어쓰기(18시 전 종가는 잠정)
python build.py         # → kospi_stage_dashboard.html (단일 파일, 오프라인)
```

## 파일
- `stage.js` 판정 엔진(브라우저·node 공용). 파라미터는 맨 위 `P`.
- `template.html` 화면. `build.py`가 stage.js와 데이터를 인라인한다.
- `test_stage.js` node로 구간·단계별 이후 수익률 출력 (`node test_stage.js [confirm일수]`).

## 의존성
`fetch_kospi.py`는 상위 폴더의 `kis_api/kis_client.py`(한투 Open API 클라이언트, 이 레포에는 없음)를 쓴다.
앱키는 환경변수 `hts_api` / `hts_api_secret`에서 읽는다. 키 없이도 `data/kospi_daily.csv`로 `python build.py`는 돈다.
