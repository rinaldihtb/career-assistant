import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_SQLITE_PATH = os.environ["DB_SQLITE_PATH"]
TABLE_NAME = os.environ["DB_TABLE_NAME"]

BATCH_SIZE = int(os.getenv("SEED_BATCH_SIZE", "100"))
CHUNK_SIZE = int(os.getenv("SEED_CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("SEED_CHUNK_OVERLAP", "50"))
MAX_ATTEMPTS = int(os.getenv("SEED_MAX_ATTEMPTS", "3"))
MAX_ATTEMPTS = int(os.getenv("SEED_MAX_ATTEMPTS", "3"))
RETRY_DELAY = int(os.getenv("SEED_RETRY_DELAY", "2"))


def seed_sqlite(dataset):
    with sqlite3.connect(Path(DB_SQLITE_PATH)) as db:
        db.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        db.execute(f"""
            CREATE TABLE {TABLE_NAME} (
                id INTEGER PRIMARY KEY,
                job_title VARCHAR(255) NOT NULL,
                company_name VARCHAR(255),
                location VARCHAR(255),
                work_type VARCHAR(50),
                job_description TEXT,
                salary_min INTEGER,
                salary_max INTEGER,
                salary_period VARCHAR(20),
                salary_currency VARCHAR(3),
                status VARCHAR(30),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        for start in range(0, len(dataset), BATCH_SIZE):
            batch = dataset.iloc[start : start + BATCH_SIZE]

            rows = [
                (
                    row.id,
                    row.job_title,
                    row.company_name,
                    row.location,
                    row.work_type,
                    row.job_description,
                    row.salary_min,
                    row.salary_max,
                    row.salary_period,
                    row.salary_currency,
                    row.status,
                )
                for row in batch.itertuples(index=False)
            ]

            db.executemany(
                f"""
                INSERT INTO {TABLE_NAME}
                (
                    id, job_title, company_name, location, work_type,
                    job_description, salary_min, salary_max, salary_period,
                    salary_currency, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        db.commit()


def main() -> None:
    """Load the cleaned dataset and seed it into SQLite."""

    dataset_path = Path("data/dataset.json")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_path}")

    start_time = time.perf_counter()
    dataset = pd.read_json(dataset_path)

    seed_sqlite(dataset)

    elapsed_time = time.perf_counter() - start_time
    print(f"Seeded {len(dataset)} rows in {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()
