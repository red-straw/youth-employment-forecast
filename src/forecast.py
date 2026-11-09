"""Step 4. Prophet + 경제지표 Regressor 모델로 2025년 고용률 백테스트.

2010-01 ~ 2024-12로 학습하고, 2025년 실제 경제지표를 넣어 2025년 고용률을 예측한 뒤
실제값과 비교(MAE). 결과 그래프는 results/figures/에 저장.
"""
import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

from common import FIG_DIR, ROOT, TARGET, load_dataset, setup_korean_font
from select_features import SELECTED_FEATURES_FILE

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
SPLIT_DATE = "2025-01-01"


def plot(train, test, forecast, mae, zoom=False):
    plt.figure(figsize=(12, 5.5))
    if not zoom:
        plt.plot(train["ds"], train["y"], label="학습 구간 (~2024)", color="gray", alpha=0.6)
    plt.plot(test["ds"], test["y"], label="2025 실제값", color="#1f77b4", linewidth=2.5, marker="s" if zoom else None)
    plt.plot(forecast["ds"], forecast["yhat"], label="2025 예측값", color="#d62728", linestyle="--", linewidth=2, marker="o")
    if zoom:
        plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="#d62728", alpha=0.1,
                         label="예측 구간")
        plt.xticks(rotation=45)
    title = "2025년 청년 고용률 실제 vs 예측" if zoom else "청년(20~29세) 고용률 추이와 2025년 예측"
    plt.title(f"{title} (MAE {mae:.3f}%p)", fontsize=14, fontweight="bold")
    plt.ylabel("고용률 (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / ("forecast_2025_zoom.png" if zoom else "forecast_all.png"), dpi=150)
    plt.close()


def main():
    features = json.loads(SELECTED_FEATURES_FILE.read_text(encoding="utf-8"))
    df = load_dataset().reset_index().rename(columns={"Date": "ds", TARGET: "y"})
    train, test = df[df["ds"] < SPLIT_DATE], df[df["ds"] >= SPLIT_DATE]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for f in features:
        model.add_regressor(f)
    model.fit(train[["ds", "y", *features]])
    forecast = model.predict(test[["ds", *features]])

    mae = mean_absolute_error(test["y"], forecast["yhat"])
    result = pd.DataFrame({"Date": test["ds"].dt.strftime("%Y-%m").values, "Actual": test["y"].values,
                           "Predicted": forecast["yhat"].round(2).values})
    result["Error"] = (result["Actual"] - result["Predicted"]).round(2)
    print(result.to_string(index=False))
    print(f"\n변수 {features}\nMAE: {mae:.4f}%p")

    setup_korean_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot(train, test, forecast, mae)
    plot(train, test, forecast, mae, zoom=True)
    result.to_csv(ROOT / "results" / "forecast_2025.csv", index=False)


if __name__ == "__main__":
    main()
