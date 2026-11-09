# Youth Employment Forecast

경제지표로 20대 청년 고용률에 영향을 주는 변수를 고르고, Prophet 모델로 2025년 고용률을 예측해 실제값과 비교한 데이터 분석 프로젝트입니다.

25-2, 개인 프로젝트

![forecast](results/figures/forecast_2025_zoom.png)

## 결과

- 경제지표 10개 중 상관계수, VIF, p-value 순으로 걸러 변수 7개를 골랐습니다. (OLS R² 0.77)
- 2010~2024년 데이터로 학습한 Prophet + Regressor 모델의 2025년 1~9월 MAE는 0.37%p입니다.
- 1차 시도(지표를 먼저 예측한 뒤 OLS로 고용률 예측)는 결과가 좋지 않아 모델을 바꿨습니다. [v1 기록](experiments/v1_ols_failed/)

## 분석 과정

1. 데이터 수집 (2010-01 ~ 2025-09, 월별 189개)

| 변수 | 설명 | 출처 |
|---|---|---|
| `Job_Rate` (타겟) | 20~29세 고용률 (%) | 통계청 KOSIS |
| `GDP` | 실질 국내총생산 (분기, 월로 보간) | 한국은행 ECOS |
| `IPI` | 전산업생산지수 | 통계청 KOSIS |
| `CCSI` | 소비자심리지수 | 한국은행 ECOS |
| `BSI` | 기업경기실사지수 | 한국은행 ECOS |
| `EXPORT` | 수출금액지수 | 한국은행 ECOS |
| `RATE` | 기준금리 | 한국은행 ECOS |
| `PPI`, `CPI` | 생산자/소비자 물가지수 | 한국은행 ECOS, KOSIS |
| `KOSPI` | 코스피 월평균 (크롤링 수집) | 네이버 금융 |
| `DOLLAR` | 원/달러 환율 | 한국은행 ECOS |

2. 전처리: 결측 제거, 표준화(StandardScaler)

3. 변수 선정

| 단계 | 방법 | 결과 |
|---|---|---|
| 상관 분석 | 고용률과의 피어슨 상관계수 | PPI 0.72, DOLLAR 0.69, CPI 0.60 순 |
| 다중공선성 제거 | VIF 10 초과 변수 반복 제거 | CPI(182), IPI(121), PPI(28) 제거 |
| 유의성 검증 | OLS p-value 0.05 초과 변수 반복 제거 | 7개 모두 유의: GDP, CCSI, BSI, EXPORT, RATE, KOSPI, DOLLAR |

<img src="results/figures/feature_impact.png" width="70%">

4. 모델링: Prophet(추세, 연간 계절성) + 변수 7개를 Regressor로 추가해 2024년까지 학습하고, 2025년 지표를 넣어 고용률을 예측했습니다. 2025년 지표는 실제값을 사용했습니다.

| 월 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| 실제 (%) | 59.7 | 59.0 | 59.5 | 60.3 | 61.7 | 61.0 | 61.3 | 60.5 | 60.7 |
| 예측 (%) | 59.45 | 59.23 | 59.33 | 60.20 | 61.00 | 60.60 | 60.92 | 60.31 | 59.80 |

## 구조

```text
.
├── data/
│   ├── raw/                  # 지표별 데이터 (KOSPI는 월평균)
│   └── final_dataset.csv     # 월별 통합 데이터셋
├── src/
│   ├── collect/
│   │   ├── kospi.py          # KOSPI 일별 시세 수집, 월평균 변환
│   │   └── gdp_monthly.py    # 분기 GDP를 월별로 보간
│   ├── preprocess.py         # 표준화
│   ├── select_features.py    # 상관/VIF/p-value, 계수 그래프
│   ├── forecast.py           # Prophet 예측, 그래프
│   └── common.py
├── results/
├── experiments/v1_ols_failed/
└── docs/presentation.pptx
```

## 실행

```bash
pip install -r requirements.txt
cd src
python preprocess.py
python select_features.py
python forecast.py
```

## 정리 내역

포트폴리오로 올리면서 다음을 정리했습니다.

- 분석 스크립트를 `src/`로 모으고 고정 경로를 상대 경로로 바꿨습니다.
- 계수 그래프가 값을 직접 입력해 그리던 방식을 OLS 결과로 그리도록 바꿨습니다. 결과 값은 원래와 같습니다.
- 데이터 열 이름 `NSI`(실제로는 수출금액지수)를 `EXPORT`로, `DOLLER`를 `DOLLAR`로 바꿨습니다.

## 기술 스택

Python, pandas, statsmodels, scikit-learn, Prophet, matplotlib, requests

## 데이터 출처 및 라이선스

- 경제지표: 한국은행 경제통계시스템(ECOS), 통계청 국가통계포털(KOSIS)
- KOSPI: `src/collect/kospi.py`로 수집해 월평균으로 가공한 값(`data/raw/kospi.csv`)만 포함했고, 일별 원본은 포함하지 않았습니다.
- 코드: [MIT](LICENSE)
