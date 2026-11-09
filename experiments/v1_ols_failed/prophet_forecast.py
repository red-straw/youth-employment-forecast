import pandas as pd
from prophet import Prophet
import os

# 1. 데이터 로드
file_path = 'standardized_dataset.csv'
if not os.path.exists(file_path):
    print("[Error] 파일을 찾을 수 없습니다.")
    exit()

df = pd.read_csv(file_path)

# 날짜 형식 변환
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
else:
    print("[Error] 'Date' 컬럼이 필요합니다.")
    exit()

# 2. 학습 대상 변수 설정 (Top 4)
target_features = ['GDP', 'KOSPI', 'DOLLER', 'RATE']
future_periods = 12  # 2023년 12개월 예측

print(f"--- [Step 4] Prophet Forecast Start ---")
print(f"Features: {target_features}")
print("Training Range: 2010-01 ~ 2022-12")
print("Forecast Range: 2023-01 ~ 2023-12")

# 결과를 저장할 데이터프레임 생성
forecast_result = pd.DataFrame()
# 날짜 생성 (2023-01-01 ~ 2023-12-01)
future_dates = pd.date_range(start='2023-01-01', periods=12, freq='MS')
forecast_result['Date'] = future_dates

# 3. 변수별 예측 루프
for feature in target_features:
    print(f"\n>> Forecasting: {feature}...")

    # Prophet 형식으로 데이터 준비 (ds, y)
    # 2022년 12월까지의 데이터만 학습에 사용 (2023년은 미지의 영역)
    train_df = df[df['Date'] < '2023-01-01'][['Date', feature]].rename(columns={'Date': 'ds', feature: 'y'})

    # 모델 학습
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(train_df)

    # 2023년 예측을 위한 데이터프레임 생성
    future = model.make_future_dataframe(periods=12, freq='MS')
    forecast = model.predict(future)

    # 2023년 데이터만 추출하여 결과에 저장
    # Prophet 결과 중 마지막 12개가 2023년 예측값임
    forecast_2023 = forecast.tail(12)['yhat'].values
    forecast_result[feature] = forecast_2023

# 4. 결과 저장
output_file = '2023_prophet_X.csv'
forecast_result.to_csv(output_file, index=False)

print("\n" + "=" * 40)
print(f"[Complete] Prediction Saved: {output_file}")
print("=" * 40)
print(forecast_result)