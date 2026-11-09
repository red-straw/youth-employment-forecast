"""Step 3. 핵심 변수 선정 (3단계 필터링) + 영향력 시각화

3-1. 상관계수로 고용률과의 영향력 순위 파악
3-2. VIF > 10 변수를 하나씩 제거해 다중공선성 해소
3-3. OLS p-value > 0.05 변수를 하나씩 제거해 유의한 변수만 남김
3-4. 최종 변수의 OLS 계수를 막대그래프로 저장
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import FIG_DIR, ROOT, STANDARDIZED_DATASET, TARGET, load_dataset, setup_korean_font

VIF_LIMIT = 10
P_LIMIT = 0.05
SELECTED_FEATURES_FILE = ROOT / "results" / "selected_features.json"


def rank_by_correlation(df):
    corr = df.corr()[TARGET].drop(TARGET)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def filter_by_vif(X):
    X = X.copy()
    while True:
        vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                        index=X.columns).sort_values(ascending=False)
        if vif.iloc[0] < VIF_LIMIT:
            return X, vif
        print(f"  [VIF 제거] {vif.index[0]} ({vif.iloc[0]:.1f})")
        X = X.drop(columns=vif.index[0])


def filter_by_pvalue(X, y):
    X = X.copy()
    while True:
        model = sm.OLS(y, sm.add_constant(X)).fit()
        pvalues = model.pvalues.drop("const")
        if pvalues.max() <= P_LIMIT:
            return X, model
        print(f"  [p-value 제거] {pvalues.idxmax()} ({pvalues.max():.4f})")
        X = X.drop(columns=pvalues.idxmax())


def plot_coefficients(coefs: pd.Series):
    setup_korean_font()
    coefs = coefs.reindex(coefs.abs().sort_values().index)
    plt.figure(figsize=(10, 5))
    bars = plt.barh(coefs.index, coefs.values,
                    color=["#d62728" if v > 0 else "#1f77b4" for v in coefs.values], alpha=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.02 if w > 0 else w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}",
                 va="center", ha="left", fontsize=10)
    plt.title("청년 고용률 결정 요인 (표준화 OLS 계수)", fontsize=14, fontweight="bold")
    plt.xlabel("계수 (+: 고용률 상승 / -: 하락)")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "feature_impact.png", dpi=150)


def main():
    df = load_dataset(STANDARDIZED_DATASET)
    X, y = df.drop(columns=TARGET), df[TARGET]

    print("[3-1] 고용률과의 상관계수 순위")
    print(rank_by_correlation(df).round(4).to_string(), "\n")

    print(f"[3-2] VIF 필터 (시작 변수 {X.shape[1]}개)")
    X, vif = filter_by_vif(X)
    print(vif.round(2).to_string(), "\n")

    print(f"[3-3] p-value 필터 (시작 변수 {X.shape[1]}개)")
    X, model = filter_by_pvalue(X, y)
    features = list(X.columns)
    print(f"최종 변수 {len(features)}개: {features}  (R² = {model.rsquared:.3f})\n")
    print(model.summary())

    SELECTED_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_FEATURES_FILE.write_text(json.dumps(features), encoding="utf-8")
    plot_coefficients(model.params.drop("const"))
    print(f"\n저장: {SELECTED_FEATURES_FILE}, {FIG_DIR / 'feature_impact.png'}")


if __name__ == "__main__":
    main()
