# v1: 경제지표 예측 후 OLS로 고용률 예측

1. 변수 4개(GDP, KOSPI, DOLLAR, RATE)를 Prophet으로 각각 2023년까지 예측했습니다. (`prophet_forecast.py`)
2. 2010~2022년 데이터로 학습한 OLS 회귀식에 예측한 지표를 넣어 2023년 고용률을 예측했습니다. (`ols_predict.py`)

결과: 표준화 값 기준 MAE 1.66으로, 2023년 실제 고용률 흐름과 맞지 않았습니다. (`final_result_comparison.csv`)

이후 변수 선정에 p-value 검증을 추가하고, 고용률을 Prophet + Regressor로 직접 모델링하는 방식으로 바꿨습니다.

파일은 당시 코드 그대로 두었습니다. 실행하려면 당시의 train/test csv가 필요합니다.
