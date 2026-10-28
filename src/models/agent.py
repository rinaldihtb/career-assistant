"""Kontrak output bersama untuk semua agent (RAG, SQL, dan supervisor).

Supervisor, UI, dan script uji hanya perlu mengenal satu bentuk hasil ini,
apa pun agent yang menjawab.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Harga standar OpenAI dalam USD per 1 juta token: (input, output). Nama model dari API memuat
# tanggal snapshot ("gpt-4o-mini-2024-07-18"), jadi dicocokkan dengan awalan.
MODEL_PRICES_USD = {"gpt-4o-mini": (0.15, 0.60)}


class LLMCall(BaseModel):
    """Satu panggilan LLM: langkah mana, model apa, dan berapa token."""

    step: str
    model: str = "-"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost_usd(self) -> float | None:
        """Biaya panggilan ini dari ``MODEL_PRICES_USD``; ``None`` jika harga model tidak ada."""
        for name, (input_price, output_price) in MODEL_PRICES_USD.items():
            if self.model.startswith(name):
                return (self.input_tokens * input_price + self.output_tokens * output_price) / 1e6
        return None


class LLMUsage(BaseModel):
    """Akumulasi pemakaian LLM selama satu kali agent berjalan.

    Total disimpan terpisah dari ``calls`` agar kode lama yang hanya membaca
    total (sample script, test) tidak perlu berubah.
    """

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: list[LLMCall] = Field(default_factory=list)

    @property
    def cost_usd(self) -> float | None:
        """Total biaya semua panggilan; ``None`` jika ada model tanpa harga atau rincian kurang."""
        costs = [call.cost_usd for call in self.calls]
        if None in costs or len(costs) != self.llm_calls:
            return None
        return sum(costs)

    def add(self, message: Any, step: str = "llm") -> None:
        """Catat satu panggilan LLM dari pesan balasan model.

        Args:
            message: Balasan LLM (biasanya ``AIMessage``). Token dibaca dari
                ``usage_metadata`` dan nama model dari ``response_metadata``
                (OpenAI mengisi ``model_name``, misalnya "gpt-4o-mini-2024-07-18").
                LLM palsu saat test tidak punya keduanya, jadi token 0 dan model "-".
            step: Nama langkah untuk ditampilkan di UI, misalnya "router" atau "sql_answer".
        """
        usage = getattr(message, "usage_metadata", None) or {}
        metadata = getattr(message, "response_metadata", None) or {}
        call = LLMCall(
            step=step,
            model=metadata.get("model_name") or metadata.get("model") or "-",
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        self.calls.append(call)
        self.llm_calls += 1
        self.input_tokens += call.input_tokens
        self.output_tokens += call.output_tokens
        self.total_tokens += call.total_tokens

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
            calls=[*self.calls, *other.calls],
        )


class AgentResult(BaseModel):
    """Hasil akhir satu agent untuk satu pertanyaan.

    Attributes:
        answer: Teks jawaban untuk pengguna.
        source: Agent yang menghasilkan jawaban. ``"general"`` berarti supervisor
            menjawab sendiri dengan pesan tetap (sapaan atau di luar topik).
            ``"multi"`` berarti jawaban gabungan beberapa sub-tugas; isinya ada di ``parts``.
            ``"cv"`` berarti jawaban dari hasil Cocokkan CV di sesi, tanpa LLM.
        job_ids: ID lowongan yang dikutip (RAG) atau muncul di hasil query (SQL).
        sql: Query yang akhirnya dieksekusi (khusus SQL agent).
        rows: Data mentah hasil query atau lowongan yang ditemukan, untuk UI dan debug.
        usage: Jumlah panggilan LLM dan token.
        error: Pesan error internal jika agent gagal; ``None`` jika sukses.
        parts: Khusus ``source="multi"``: satu ``AgentResult`` per bagian jawaban
            (per sub-tugas, ditambah catatan jika ada), urut tampil di UI. Kosong
            untuk hasil satu agent, sehingga kode lama tidak terpengaruh.
    """

    answer: str
    source: Literal["rag", "sql", "cv", "data", "general", "multi"]
    job_ids: list[int] = Field(default_factory=list)
    sql: str | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    error: str | None = None
    parts: list["AgentResult"] = Field(default_factory=list)
