"""RAG agent: mencari lowongan relevan lalu menjawab dengan sitasi [ID:x].

Alur ``retrieve_jobs`` (tanpa LLM chat, hanya embedding query):
    1. Deteksi lokasi dari kolom ``location`` melalui ``JobRepository``; lokasi
       setelah "selain/kecuali/di luar/bukan" menjadi pengecualian.
    2. Cari chunk di Qdrant; bila ada lokasi, filter ``metadata.job_id`` (lowongan
       di lokasi diminta dikurangi lokasi dikecualikan) dan buang nama tempat dari
       query embedding.
    3. Buang chunk pendek, dedupe berdasarkan ``job_id``, lalu ambil kandidat
       dengan skor tertinggi.
    4. Ambil detail lowongan dari repository berdasarkan ID.

Hasil retrieval dikirim sebagai tool artifact. LLM RAG kemudian menjawab hanya
berdasarkan kandidat tersebut.
"""

import re
from collections.abc import Iterable
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from qdrant_client import models

from src.agents.prompts import RAG_ANSWER_SYSTEM
from src.repositories.job_repository import JobRepository

TOP_N = 5
CANDIDATE_K = 20
SCORE_THRESHOLD = 0.45
JOB_ID_KEY = "metadata.job_id"
MIN_CHUNK_CHARS = 40
GENERIC_PLACE_WORDS = {
    "selatan",
    "utara",
    "barat",
    "timur",
    "pusat",
    "tengah",
    "raya",
    "kota",
    "kabupaten",
    "district",
    "regency",
    "south",
    "north",
    "west",
    "east",
    "central",
    "lama",
    "baru",
    "di",
}
# Kata sebelum nama tempat yang membalik maknanya. Tanpa ini "sales selain di Jakarta"
# memfilter kandidat ke Jakarta dan jawabannya "tidak ada lowongan" (T14-02).
EXCLUSION_PREFIX = r"(?:selain|kecuali|di\s+luar|bukan)\s+(?:(?:di|dari)\s+)?"


class RetrievalResult(BaseModel):
    """Hasil pencarian lowongan sebelum dijawab LLM.

    Attributes:
        places: Lokasi yang diminta pada pertanyaan.
        excluded_places: Lokasi yang dikecualikan ("selain Jakarta" -> ``["jakarta"]``).
        jobs: Lowongan terpilih, urut skor tertinggi, dengan ``score`` dan
            ``matched_chunk`` untuk setiap lowongan.
    """

    jobs: list[dict[str, Any]] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    excluded_places: list[str] = Field(default_factory=list)


class RagAgent:
    """Agent RAG yang mencari lowongan berdasarkan kemiripan semantik."""

    def __init__(self, llm: BaseChatModel, job_repository: JobRepository) -> None:
        self.llm = llm
        self.job_repository = job_repository

    def create_rag_agent(self) -> CompiledStateGraph:
        """Buat LangChain agent dengan tool pencarian lowongan."""
        retrieve_jobs_tool = StructuredTool.from_function(
            self.retrieve_jobs,
            name="retrieve_jobs",
            description="Cari lowongan yang relevan untuk satu pertanyaan.",
            response_format="content_and_artifact",
        )
        return create_agent(
            model=self.llm,
            system_prompt=RAG_ANSWER_SYSTEM,
            tools=[retrieve_jobs_tool],
            name="rag_worker",
        )

    def retrieve_jobs(
        self,
        question: str,
        top_n: int = TOP_N,
        k: int = CANDIDATE_K,
        score_threshold: float = SCORE_THRESHOLD,
    ) -> tuple[str, RetrievalResult]:
        """Cari lowongan relevan dan simpan hasil lengkap sebagai tool artifact.

        Jika pertanyaan menyebut lokasi, pencarian Qdrant dibatasi ke lowongan
        di lokasi yang diminta (semua lokasi jika hanya ada pengecualian) dikurangi
        lokasi yang dikecualikan. Jika tidak, kandidat dengan skor di bawah
        ``score_threshold`` dibuang. Hasilnya berupa JSON untuk LLM dan
        ``RetrievalResult`` untuk supervisor.
        """
        place_index = self._build_place_index(self.job_repository.get_job_locations())
        detected = self._find_places(question, place_index)
        places, excluded = self._split_excluded_places(question, detected)

        search_query = self._remove_places(question, detected)
        qdrant_filter = None
        if detected:
            job_ids = self._allowed_job_ids(place_index, places, excluded)
            if not job_ids:
                result = RetrievalResult(places=places, excluded_places=excluded)
                return result.model_dump_json(), result
            qdrant_filter = self._build_filter(job_ids)
        chunks = self.job_repository.similarity_search_with_score(
            search_query, k=k, filter=qdrant_filter
        )

        best_chunks: dict[int, tuple[float, str]] = {}
        for document, score in chunks:
            job_id = document.metadata.get("job_id")
            if job_id is None or len(document.page_content.strip()) < MIN_CHUNK_CHARS:
                continue
            best_chunks.setdefault(int(job_id), (float(score), document.page_content))

        ranked = sorted(best_chunks.items(), key=lambda item: item[1][0], reverse=True)
        if qdrant_filter is None:
            ranked = [item for item in ranked if item[1][0] >= score_threshold]

        selected = ranked[:top_n]
        jobs = self.job_repository.get_jobs_by_ids(job_id for job_id, _ in selected)
        chunks_by_id = dict(selected)
        for job in jobs:
            job["score"], job["matched_chunk"] = chunks_by_id[job["id"]]

        result = RetrievalResult(jobs=jobs, places=places, excluded_places=excluded)
        return result.model_dump_json(), result

    def find_places(self, question: str) -> list[str]:
        """Lokasi dari kolom ``location`` yang disebut pertanyaan, termasuk yang dikecualikan."""
        place_index = self._build_place_index(self.job_repository.get_job_locations())
        return self._find_places(question, place_index)

    def _build_place_index(self, job_locations: Iterable[tuple[int, str]]) -> dict[str, set[int]]:
        """Petakan nama atau alias lokasi ke ID lowongan."""
        index: dict[str, set[int]] = {}
        for job_id, location in job_locations:
            for name in location.split("\n", 1)[0].lower().split(","):
                name = name.strip()
                short_name = " ".join(
                    word for word in name.split() if word not in GENERIC_PLACE_WORDS
                )
                for key in {name, short_name} - {""}:
                    index.setdefault(key, set()).add(job_id)
        return index

    def _find_places(self, question: str, place_index: dict[str, set[int]]) -> list[str]:
        """Temukan lokasi yang disebut dalam pertanyaan."""
        text = question.lower()
        found: list[str] = []
        for place in sorted(place_index, key=len, reverse=True):
            pattern = self._place_pattern(place)
            if pattern.search(text):
                found.append(place)
                text = pattern.sub(" ", text)
        return found

    def _remove_places(self, question: str, places: Iterable[str]) -> str:
        """Hilangkan lokasi dari teks yang akan di-embed."""
        text = question
        for place in places:
            pattern = self._place_pattern(place)
            text = pattern.sub(" ", text)
        text = " ".join(text.split())
        return text if text.strip(" ?!.,") else question

    def _place_pattern(self, place: str) -> re.Pattern[str]:
        """Buat regex lokasi dengan batas kata."""
        words = r"\s+".join(map(re.escape, place.split()))
        return re.compile(rf"(?<!\w){words}(?!\w)", re.IGNORECASE)

    def _split_excluded_places(
        self, question: str, places: Iterable[str]
    ) -> tuple[list[str], list[str]]:
        """Pisahkan lokasi menjadi ``(diminta, dikecualikan)``.

        Lokasi dikecualikan jika langsung didahului ``EXCLUSION_PREFIX``. Regex cukup
        karena pengecualian lokasi hampir selalu memakai empat kata itu; "selain Jakarta
        dan Bandung" hanya mengecualikan Jakarta.
        """
        included: list[str] = []
        excluded: list[str] = []
        for place in places:
            words = r"\s+".join(map(re.escape, place.split()))
            pattern = re.compile(rf"(?<!\w){EXCLUSION_PREFIX}{words}(?!\w)", re.IGNORECASE)
            (excluded if pattern.search(question) else included).append(place)
        return included, excluded

    def _allowed_job_ids(
        self, place_index: dict[str, set[int]], places: list[str], excluded: list[str]
    ) -> set[int]:
        """ID lowongan di ``places`` (atau semua lokasi jika kosong) dikurangi ``excluded``."""
        allowed = set().union(*(place_index[place] for place in places or place_index))
        return allowed - set().union(*(place_index[place] for place in excluded))

    def _build_filter(self, job_ids: Iterable[int]) -> models.Filter:
        """Buat filter Qdrant yang hanya meloloskan chunk dengan ``metadata.job_id`` di daftar."""
        return models.Filter(
            must=[models.FieldCondition(key=JOB_ID_KEY, match=models.MatchAny(any=sorted(job_ids)))]
        )
