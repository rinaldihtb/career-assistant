import os
import time
import uuid

import pandas as pd
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.database import VECTOR_DB

load_dotenv()

BATCH_SIZE = int(os.getenv("SEED_BATCH_SIZE", "100"))
CHUNK_SIZE = int(os.getenv("SEED_CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("SEED_CHUNK_OVERLAP", "50"))
MAX_ATTEMPTS = int(os.getenv("SEED_MAX_ATTEMPTS", "3"))
MAX_ATTEMPTS = int(os.getenv("SEED_MAX_ATTEMPTS", "3"))
RETRY_DELAY = int(os.getenv("SEED_RETRY_DELAY", "2"))


def chunk_documents(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    documents = []
    document_ids = []

    for row in data.itertuples(index=False):
        # Input informasi Resume_id dan category pada setiap dokumen
        chunks = splitter.create_documents(
            texts=[row.job_description],
            metadatas=[
                {
                    "job_id": row.id,
                }
            ],
        )
        for chunk_index, chunk in enumerate(chunks):
            # Menambahkan informasi chunk_index ke document,
            chunk.metadata["chunk_index"] = chunk_index
            documents.append(chunk)

            # generate document_id dengan uuid unik untuk mencegah duplikasi data di vector database
            document_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{row.id}_{chunk_index}",
            )
            document_ids.append(str(document_id))

    # Array Dokuments
    # Array Dokument IDS
    return documents, document_ids


def preprocess_documents(dataset: pd.DataFrame):
    start_time = time.perf_counter()  ## Logging Purpose
    dataset_count = len(dataset)  ## Logging Purpose

    documents, document_ids = chunk_documents(dataset)
    document_count = len(documents)

    for batch_start in range(0, document_count, BATCH_SIZE):
        batch_number = (batch_start // BATCH_SIZE) + 1
        batch_end = batch_start + BATCH_SIZE
        batch_documents = documents[batch_start:batch_end]
        batch_ids = document_ids[batch_start:batch_end]

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                # Bulk Insert Vector Database
                VECTOR_DB.add_documents(
                    documents=batch_documents,
                    ids=batch_ids,
                )
                break
            except Exception as error:
                print(
                    f"Batch {batch_number} failed "
                    f"(attempt {attempt}/{MAX_ATTEMPTS}): {error}"
                )

                if attempt == MAX_ATTEMPTS:
                    raise

                time.sleep(RETRY_DELAY)

        processed_count = min(batch_end, document_count)
        progress = processed_count / document_count
        print(
            f"Processed {processed_count}/{document_count} documents ({progress:.1%})"
        )

    elapsed_seconds = time.perf_counter() - start_time
    summary = (
        f"Seeding completed: {dataset_count} dataset rows, "
        f"{document_count} documents, {elapsed_seconds:.2f} seconds"
    )

    print(summary)


if __name__ == "__main__":
    df = pd.read_json("data/dataset.json")
    preprocess_documents(df)
