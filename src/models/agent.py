"""Kontrak output bersama untuk semua agent (RAG, SQL, dan supervisor).

Supervisor, UI, dan script uji hanya perlu mengenal satu bentuk hasil ini,
apa pun agent yang menjawab.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMUsage(BaseModel):
    """Akumulasi pemakaian LLM selama satu kali agent berjalan."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, message: Any) -> None:
        """Catat satu panggilan LLM dari pesan balasan model.

        Args:
            message: Balasan LLM (biasanya ``AIMessage``). Token dibaca dari
                ``usage_metadata``. LLM palsu saat test tidak punya
                ``usage_metadata``, jadi yang bertambah hanya ``llm_calls``.
        """
        self.llm_calls += 1
        usage = getattr(message, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)

    def merged(self, other: "LLMUsage") -> "LLMUsage":
        """Gabungkan dua catatan pemakaian tanpa mengubah keduanya.

        Dipakai supervisor untuk menjumlahkan panggilan router dengan
        panggilan agent yang dipilih, sehingga UI melihat biaya total.
        """
        return LLMUsage(
            llm_calls=self.llm_calls + other.llm_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class AgentResult(BaseModel):
    """Hasil akhir satu agent untuk satu pertanyaan.

    Attributes:
        answer: Teks jawaban untuk pengguna.
        source: Agent yang menghasilkan jawaban. ``"general"`` berarti supervisor
            menjawab sendiri dengan pesan tetap (sapaan atau di luar topik).
        job_ids: ID lowongan yang dikutip (RAG) atau muncul di hasil query (SQL).
        sql: Query yang akhirnya dieksekusi (khusus SQL agent).
        rows: Data mentah hasil query atau lowongan yang ditemukan, untuk UI dan debug.
        usage: Jumlah panggilan LLM dan token.
        error: Pesan error internal jika agent gagal; ``None`` jika sukses.
    """

    answer: str
    source: Literal["rag", "sql", "general"]
    job_ids: list[int] = Field(default_factory=list)
    sql: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    error: str | None = None
