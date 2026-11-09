"""분기별 실질 GDP(data/raw/gdp_year.csv)를 월별로 보간 → data/raw/gdp.csv
분기 시작월에 값을 두고 시간 기준 선형 보간, 마지막 분기 이후는 직전 값 유지."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RAW_DIR  # noqa: E402

END_DATE = "2025-09-01"


def quarter_to_date(text):
    """'2010/1/4' 또는 '2010.1/4' 형식 → 해당 분기 첫 달."""
    text = str(text).replace("/4", "")
    year, q = text.replace("/", ".").split(".")
    return pd.Timestamp(year=int(year), month=(int(q) - 1) * 3 + 1, day=1)


def main():
    src = RAW_DIR / "gdp_year.csv"
    try:
        gdp = pd.read_csv(src, encoding="utf-8")
    except UnicodeDecodeError:
        gdp = pd.read_csv(src, encoding="cp949")

    gdp["Date"] = gdp["date"].apply(quarter_to_date)
    gdp["GDP"] = gdp["gdp"].astype(str).str.replace(",", "").astype(float)
    gdp = gdp.set_index("Date")[["GDP"]].sort_index()

    months = pd.date_range(gdp.index.min(), END_DATE, freq="MS")
    monthly = gdp.reindex(months).interpolate(method="time").ffill()
    monthly.index.name = "Date"
    monthly.to_csv(RAW_DIR / "gdp.csv", encoding="utf-8-sig")
    print(f"저장: {RAW_DIR / 'gdp.csv'} ({len(monthly)}개월)")


if __name__ == "__main__":
    main()
