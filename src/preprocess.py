"""Step 2. 결측치 제거 + 표준화(StandardScaler) -> data/standardized_dataset.csv"""
from sklearn.preprocessing import StandardScaler

from common import STANDARDIZED_DATASET, load_dataset


def main():
    df = load_dataset()
    print(f"데이터 크기: {df.shape} ({df.index.min():%Y-%m} ~ {df.index.max():%Y-%m})")

    scaled = df.copy()
    scaled[df.columns] = StandardScaler().fit_transform(df)
    scaled.to_csv(STANDARDIZED_DATASET)
    print(f"저장: {STANDARDIZED_DATASET}")


if __name__ == "__main__":
    main()
