"""KOSPI 일별 종가를 수집해 월평균으로 변환 → data/raw/kospi.csv
(출처: 네이버 금융 KOSPI 일별 시세, 2010-01 ~ 2025-09)

※ 학습 목적으로 작성한 코드입니다.
  - 서버 부담을 줄이기 위해 요청 간격을 두고(50페이지마다 3~5초 대기) 수집합니다.
  - 수집한 원본 데이터는 재배포하지 않으며, 분석에는 월평균으로 가공한 값만 사용했습니다.
  - 실행 전 대상 사이트의 이용약관과 robots.txt를 확인하세요.
"""
import random
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW_DIR  # noqa: E402

URL = "https://finance.naver.com/sise/sise_index_day.naver?code=KOSPI&page={}"
PAGES = 660
START, END = "2010-01-01", "2025-09-30"


def main():
    tables = []
    for page in range(1, PAGES + 1):
        if page % 50 == 0:  # 과도한 요청 방지
            print(f"{page}/{PAGES} 페이지")
            time.sleep(random.uniform(3, 5))
        try:
            html = requests.get(URL.format(page), headers={"user-agent": "Mozilla/5.0"}, timeout=10).text
            tables.append(pd.read_html(StringIO(html))[0])
        except Exception as e:
            print(f"[skip] {page} 페이지: {e}")

    df = pd.concat(tables).dropna()[["날짜", "체결가"]]
    df.columns = ["Date", "KOSPI"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["KOSPI"] = df["KOSPI"].astype(str).str.replace(",", "").astype(float)
    df = df[(df["Date"] >= START) & (df["Date"] <= END)].set_index("Date").sort_index()

    monthly = df.resample("MS").mean().round(2)
    monthly.to_csv(RAW_DIR / "kospi.csv", encoding="utf-8-sig")
    print(f"저장: {RAW_DIR / 'kospi.csv'} ({monthly.index.min():%Y-%m} ~ {monthly.index.max():%Y-%m})")


if __name__ == "__main__":
    main()
