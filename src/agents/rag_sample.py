from pprint import pprint

from config.database import VECTOR_DB
from langchain_community.utilities import SQLDatabase


def testing():
    results = VECTOR_DB.similarity_search_with_score(query="Marketing", k=5)
    pprint(results)

    DB = SQLDatabase.from_uri(
        "sqlite:///data/database.sqlite", engine_args={"pool_pre_ping": True}
    )

    resume_ids = [302, 175, 340, 249]
    placeholders = ",".join(map(str, resume_ids))

    query = f"""
    SELECT *
    FROM jobs
    WHERE id IN ({placeholders})
    """
    query_result = DB.run(query, include_columns=True)
    pprint(query_result)


if __name__ == "__main__":
    testing()
