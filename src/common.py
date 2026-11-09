"""공통 경로·데이터 로더."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FINAL_DATASET = DATA_DIR / "final_dataset.csv"
STANDARDIZED_DATASET = DATA_DIR / "standardized_dataset.csv"
FIG_DIR = ROOT / "results" / "figures"

TARGET = "Job_Rate"  # 20~29세 고용률 (%)


def load_dataset(path: Path = FINAL_DATASET) -> pd.DataFrame:
    """월별 통합 데이터셋을 Date 인덱스의 숫자형 DataFrame으로 로드."""
    df = pd.read_csv(path, thousands=",", index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df.apply(pd.to_numeric, errors="coerce").dropna()


def setup_korean_font():
    """그래프 한글 폰트 설정 (Windows: 맑은 고딕, 그 외: 설치된 CJK 폰트)."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
