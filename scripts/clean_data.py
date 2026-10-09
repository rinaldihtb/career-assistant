import re

import pandas as pd

SALARY_PATTERN = re.compile(
    r"Rp\s*([\d.,]+)(?:\s*[–-]\s*Rp\s*([\d.,]+))?\s*per\s*(month|year)",
    re.IGNORECASE,
)


def _to_number(value: str | None) -> int | None:
    """Convert an Indonesian-formatted salary string into an integer."""
    return int(re.sub(r"[^\d]", "", value)) if value else None


def _parse_salary(
    value,
) -> tuple[int | None, int | None, str | None, str | None]:
    """Extract minimum, maximum, period, and currency from one salary value."""
    if pd.isna(value):
        return None, None, None, None

    match = SALARY_PATTERN.search(str(value))
    if not match:
        return None, None, None, None

    salary_min = _to_number(match.group(1))
    salary_max = _to_number(match.group(2)) or salary_min
    return salary_min, salary_max, match.group(3).lower(), "IDR"


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Return job data with a one-based incremental index."""
    cleaned_data = data.copy()
    cleaned_data.index = pd.RangeIndex(start=1, stop=len(cleaned_data) + 1)
    cleaned_data.index.name = "id"

    # Ekstrak salary menjadi angka; nilai yang kosong/tidak valid menjadi None.
    salary_columns = pd.DataFrame(
        cleaned_data["salary"].map(_parse_salary).tolist(),
        columns=["salary_min", "salary_max", "salary_period", "salary_currency"],
        index=cleaned_data.index,
    )
    salary_columns = salary_columns.astype(
        {
            "salary_min": "Int64",
            "salary_max": "Int64",
        }
    )
    cleaned_data[salary_columns.columns] = salary_columns
    cleaned_data['status'] = "OPEN" # open / closed

    # Kolom salary mentah tidak lagi diperlukan setelah nilai terstrukturnya dibuat.
    cleaned_data = cleaned_data.drop(columns=["salary"])
    return cleaned_data


if __name__ == "__main__":
    # Panggil jobs.json ke dalam array
    df = pd.read_json("data/raw_dataset.json")
    cleaned_df = clean_dataset(df)
    # Store cleaned_df ke json "dataset.json"
    cleaned_df.reset_index().to_json(
        "data/dataset.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    print(cleaned_df)
