import pandas as pd
import statsmodels.api as sm
import numpy as np
import os

# 1. 파일 경로 설정
train_file = 'train_data.csv'        # 학습용
test_file = 'test_data.csv'          # 정답 확인용
prophet_file = '2023_prophet_X.csv'  # 예측 입력용

# 파일 존재 확인
if not all(os.path.exists(f) for f in [train_file, test_file, prophet_file]):
    print("[Error] 필요한 데이터 파일을 찾을 수 없습니다.")
    exit()

# 2. 데이터 로드
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)
df_prophet = pd.read_csv(prophet_file)

# 날짜 인덱스 설정
if 'Date' in df_train.columns: df_train.set_index('Date', inplace=True)
if 'Date' in df_test.columns: df_test.set_index('Date', inplace=True)

# 3. [5-1] 모델 학습 (Train Data 사용)
features = ['GDP', 'KOSPI', 'DOLLER', 'RATE']
target = 'Job_Rate'

print("--- [Step 5] Model Training & Verification ---")

# [Fix] 데이터 전처리: 결측치(NaN) 및 무한대(Inf) 제거
# 학습에 필요한 컬럼만 추출
train_subset = df_train[features + [target]].copy()

# 무한대(inf)를 NaN으로 변환 후 삭제
train_subset.replace([np.inf, -np.inf], np.nan, inplace=True)
original_len = len(train_subset)
train_subset.dropna(inplace=True)
cleaned_len = len(train_subset)

print(f"Training Data: {train_file} (Original: {original_len} -> Cleaned: {cleaned_len} rows)")
print(f"Features: {features}")

if cleaned_len == 0:
    print("[Critical Error] 학습할 데이터가 없습니다. 전처리 과정을 확인하세요.")
    exit()

X_train = train_subset[features]
y_train = train_subset[target]
X_train_const = sm.add_constant(X_train)

# OLS 모델 학습
model = sm.OLS(y_train, X_train_const).fit()

# 4. [5-2] 예측 실행 (Prophet 예측값 사용)
X_pred = df_prophet[features].copy()
X_pred_const = sm.add_constant(X_pred, has_constant='add')

y_pred = model.predict(X_pred_const)

# 5. [5-3] 평가 (Test Data의 실제 Job_Rate와 비교)
real_y = df_test[target].values

# 데이터 개수 맞추기
min_len = min(len(real_y), len(y_pred))
real_y = real_y[:min_len]
y_pred = y_pred.values[:min_len]
dates = df_prophet['Date'].iloc[:min_len]

# 결과 정리
results = pd.DataFrame({
    'Date': dates,
    'Actual_Job_Rate': real_y,
    'Predicted_Job_Rate': y_pred,
    'Diff': real_y - y_pred
})

# 오차율(MAE) 계산
mae = np.mean(np.abs(results['Diff']))

print("\n=== [Final Verification Results: 2023] ===")
print(results)
print("\n" + "="*40)
print(f"Final MAE (Mean Absolute Error): {mae:.4f}")
print("="*40)

# 결과 저장
results.to_csv('final_result_comparison.csv', index=False)