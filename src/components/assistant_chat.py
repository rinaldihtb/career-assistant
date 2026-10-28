"""Halaman Asisten Karier: chat yang memanggil supervisor agent lewat FastAPI.

Komponen ini hanya mengurus tampilan dan riwayat chat. Pertanyaan dan riwayat dikirim
ke ``POST /chat-assistant`` (``src/routes/routes.py``), yang menjalankan graph
``SupervisorAgent``; UI tidak memanggil agent secara langsung. Jawaban gabungan
(``source="multi"``) ditampilkan per bagian dengan aturan yang sama seperti jawaban
satu agent.

CV dibaca dari ``st.session_state["cv"]`` yang diisi halaman Cocokkan CV. Jika ada, body API
membawa profil ringkas dan hasil Cocokkan CV (``cv_payload``); nama, kontak, lokasi, riwayat
kerja, pencapaian, dan teks CV tidak dikirim. Banner di atas chat menunjukkan apakah CV dipakai.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests
import streamlit as st

from src.agents.cv_agent import profile_label
from src.components.job_description import description_to_markdown, escape_markdown
from src.components.navigation import CV_PAGE
from src.containers import get_job_repository
from src.cv.ingest import CV_STATE_KEY
from src.helpers.format import format_salary
from src.models.agent import AgentResult

logger = logging.getLogger(__name__)

# Prefix "assistant_" agar tidak bentrok dengan key session_state halaman lain.
MESSAGES_KEY = "assistant_messages"
PENDING_KEY = "assistant_pending_question"
EXAMPLE_QUESTIONS = (
    "Lowongan data analyst di Jakarta",
    "Berapa rata-rata gaji lowongan di Surabaya?",
    "Saya lulusan akuntansi, lowongan apa yang cocok?",
)
# Satu contoh per jenis jawaban saat CV aktif: agent CV (lowongan, skill), lalu SQL dengan
# posisi tujuan dari profil.
CV_EXAMPLE_QUESTIONS = (
    "Lowongan mana yang paling cocok dengan CV-ku?",
    "Skill apa yang perlu kupelajari dulu?",
    "Berapa rata-rata gaji untuk posisi yang kutuju?",
)
NO_CV_BANNER = (
    "Belum ada CV, jadi jawaban hanya dari data lowongan. Upload CV di Cocokkan CV agar "
    'Asisten bisa menjawab "lowongan yang cocok untukku" dan "skill apa yang kurang".'
)
CV_BANNER = (
    "CV aktif: {file_name} · {label}. Profil ringkas (posisi tujuan, pengalaman, "
    "pendidikan, skill) dipakai untuk menjawab; teks CV tidak dikirim."
)
# Field profil yang dibaca planner (``format_profile``) dan agent CV (``profile_label``).
CV_PROFILE_FIELDS = frozenset(
    {"target_role", "experience_months", "education_level", "education_major", "skills"}
)
API_ERROR = "Asisten belum bisa dihubungi. Silakan coba lagi."
# Alamat FastAPI dibaca dari env saat dipanggil: docker-compose mengisinya dengan nama
# layanan (http://fastapi:8000); tanpa env, `make api` di mesin yang sama.
API_URL_ENV = "ASSISTANT_API_URL"
DEFAULT_API_URL = "http://localhost:8000"
# Dikirim sebagai header X-API-Key jika terisi; nilainya sama dengan env API.
API_KEY_ENV = "ASSISTANT_API_KEY"
# (connect, read) detik; jawaban terlama di set eval 14 detik.
API_TIMEOUT = (5, 90)
SOURCE_LABELS = {
    "rag": "Pencarian lowongan (RAG)",
    "sql": "Statistik data (SQL)",
    "general": "Pesan umum",
    "multi": "Gabungan beberapa agent",
    "cv": "Hasil Cocokkan CV",
}
STEP_LABELS = {
    "planner": "Planner: pecah pertanyaan",
    "sql_generate": "SQL: tulis query",
    "sql_retry": "SQL: perbaiki query",
    "sql_answer": "SQL: rangkum hasil",
    "rag_answer": "RAG: jawab dari kandidat",
}
# Label catatan di kartu: alasan relevansi (RAG), fakta dari query (SQL, misalnya gaji), atau
# skor dan skill yang cocok (CV).
NOTE_LABELS = {"rag": "Kenapa cocok", "sql": "Keterangan", "cv": "Kecocokan"}
# SQL bisa mengembalikan sampai 50 baris; kartu dibatasi agar halaman tetap ringkas.
MAX_CARDS = 10

# Baris jawaban berformat "- [ID:123] Judul - Perusahaan (Lokasi): catatan" (prompt RAG dan SQL).
_ANSWER_JOB_LINE = re.compile(r"^\s*[-*•]?\s*\[ID:\s*(\d+)\](.*)$", re.IGNORECASE)


@dataclass
class Display:
    """Isi bubble satu jawaban, dihitung sekali lalu disimpan di riwayat.

    Disimpan agar rerun Streamlit tidak membaca ulang database untuk kartu SQL.
    ``parts`` berisi tampilan per bagian untuk jawaban gabungan, urut sama
    dengan ``AgentResult.parts``.
    """

    text: str
    cards: list[tuple[dict[str, Any], str | None]] = field(default_factory=list)
    parts: list["Display"] = field(default_factory=list)


def render_assistant_page() -> None:
    """Render halaman chat lengkap: riwayat, contoh pertanyaan, dan input."""
    st.markdown("## Asisten Karier")
    st.caption("Tanyakan lowongan atau statistik lowongan dari data kami.")

    cv: dict[str, Any] | None = st.session_state.get(CV_STATE_KEY)
    _render_cv_banner(cv)

    messages: list[dict[str, Any]] = st.session_state.setdefault(MESSAGES_KEY, [])
    for message in messages:
        _render_message(message)

    question = st.chat_input("Tanya tentang lowongan...") or st.session_state.pop(PENDING_KEY, None)
    if question:
        _answer(question, messages, cv)
    elif not messages:
        _render_examples(CV_EXAMPLE_QUESTIONS if cv else EXAMPLE_QUESTIONS)

    # Dirender terakhir agar langsung muncul setelah jawaban pertama.
    if messages:
        st.button("Mulai percakapan baru", on_click=_clear_messages)


# --- Alur tanya-jawab --------------------------------------------------------


def _answer(question: str, messages: list[dict[str, Any]], cv: dict[str, Any] | None) -> None:
    """Kirim pertanyaan ke API, tampilkan jawaban, lalu simpan ke riwayat.

    ``history`` diambil sebelum pertanyaan ditambahkan agar pertanyaan tidak terkirim
    dua kali. ``AgentResult`` dan durasi ikut disimpan agar kartu dan detail LLM tetap
    tampil saat rerun. Jika API gagal, pesan error tampil dan pertanyaan tetap di riwayat.
    CV aktif dibaca saat bertanya, jadi koreksi profil di Cocokkan CV langsung berlaku.
    """
    history = [{"role": message["role"], "content": message["content"]} for message in messages]
    user_message = {"role": "user", "content": question}
    messages.append(user_message)
    _render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            start = time.perf_counter()
            try:
                result = ask_assistant_api(history, user_message, cv_payload(cv))
            except (requests.RequestException, KeyError, TypeError, ValueError):
                logger.exception("Gagal memanggil Chat Assistant API")
                st.error(API_ERROR)
                return
            seconds = time.perf_counter() - start
            display = prepare_display(result)
        _render_result(result, display, seconds)
    if result.error:
        logger.warning("Agent %s error: %s", result.source, result.error)
    messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "result": result,
            "display": display,
            "seconds": seconds,
        }
    )


def ask_assistant_api(
    history: list[dict[str, str]],
    user_message: dict[str, str],
    cv: dict[str, Any] | None = None,
) -> AgentResult:
    """Kirim satu giliran chat ke ``POST /chat-assistant`` dan kembalikan ``AgentResult``-nya.

    ``cv`` adalah hasil ``cv_payload`` (``None`` tanpa CV). Respons API berbentuk
    ``{"result": AgentResult.model_dump(mode="json")}``, jadi job_ids, rows, parts, dan usage
    ikut kembali untuk kartu dan detail LLM.

    Raises:
        requests.RequestException: API tidak terjangkau, timeout, atau status bukan 2xx.
        KeyError, TypeError, ValueError: Isi respons tidak sesuai kontrak (termasuk
            ``pydantic.ValidationError``).
    """
    base_url = os.getenv(API_URL_ENV) or DEFAULT_API_URL
    api_key = os.getenv(API_KEY_ENV)
    response = requests.post(
        f"{base_url.rstrip('/')}/chat-assistant",
        json={"history": history, "user_message": user_message, "cv": cv},
        headers={"X-API-Key": api_key} if api_key else {},
        timeout=API_TIMEOUT,
    )
    response.raise_for_status()
    return AgentResult.model_validate(response.json()["result"])


def _ask(question: str) -> None:
    """Callback tombol contoh: pertanyaan diproses pada rerun berikutnya."""
    st.session_state[PENDING_KEY] = question


def _clear_messages() -> None:
    """Callback tombol percakapan baru."""
    st.session_state[MESSAGES_KEY] = []


# --- Tampilan ----------------------------------------------------------------


def _render_cv_banner(cv: dict[str, Any] | None) -> None:
    """Status CV di atas chat dan tombol ke Cocokkan CV (0 LLM)."""
    if cv:
        st.success(cv_banner(cv))
        clicked = st.button("Koreksi profil", key="assistant_to_cv")
    else:
        st.info(NO_CV_BANNER)
        clicked = st.button("Upload CV", key="assistant_to_cv")
    if clicked:
        st.switch_page(CV_PAGE)


def _render_examples(questions: tuple[str, ...]) -> None:
    """Tombol contoh pertanyaan saat riwayat masih kosong.

    Lebar tombol mengikuti teks dan baris boleh turun, karena di kolom berlebar tetap teks
    contoh terpotong pada lebar 1000 px.
    """
    st.markdown("**Contoh pertanyaan:**")
    with st.container(horizontal=True):
        for index, question in enumerate(questions):
            st.button(question, on_click=_ask, args=(question,), key=f"assistant_example_{index}")


def _render_message(message: dict[str, Any]) -> None:
    """Satu bubble chat dari riwayat."""
    with st.chat_message(message["role"]):
        result = message.get("result")
        if result is None:
            st.markdown(message["content"])
        else:
            _render_result(result, message["display"], message.get("seconds"))


def _render_result(result: AgentResult, display: Display, seconds: float | None) -> None:
    """Isi bubble asisten: setiap bagian jawaban, lalu satu detail LLM untuk total.

    Jawaban gabungan dirender per bagian dengan ``_render_body`` yang sama,
    sehingga kartu SQL dan RAG tetap memakai aturan masing-masing.
    """
    if result.parts:
        for part, part_display in zip(result.parts, display.parts, strict=True):
            _render_body(part, part_display)
    else:
        _render_body(result, display)
    _render_llm_usage(result, seconds)


def _render_body(result: AgentResult, display: Display) -> None:
    """Teks, kartu lowongan, dan tabel SQL untuk satu jawaban agent.

    Kartu dipakai untuk setiap lowongan yang disebut, baik dari RAG maupun SQL,
    sehingga tampilannya sama apa pun agent yang menjawab.
    """
    if display.text:
        st.markdown(display.text.replace("$", r"\$"))
    note_label = NOTE_LABELS.get(result.source, "Keterangan")
    for job, note in display.cards:
        _render_job_card(job, note, note_label)
    if result.source == "sql" and result.rows:
        with st.expander(f"Data hasil query ({len(result.rows)} baris)"):
            st.dataframe(result.rows, hide_index=True)
            if result.sql:
                st.code(result.sql, language="sql")


def _render_job_card(job: dict[str, Any], note: str | None, note_label: str) -> None:
    """Kartu lowongan dengan susunan yang sama untuk setiap lowongan."""
    location = " ".join((job.get("location") or "-").split())
    salary = format_salary(job.get("salary_min"), job.get("salary_max"))
    with st.container(border=True):
        st.markdown(
            f"**{escape_markdown(job.get('job_title') or '-')}**  \n"
            f"{escape_markdown(job.get('company_name') or '-')}"
        )
        st.caption(
            f"Lokasi: {location} | Tipe: {job.get('work_type') or '-'} | "
            f"Gaji: {salary} | ID {job['id']}"
        )
        if note:
            st.markdown(f"**{note_label}:** {escape_markdown(note)}")
        description = job.get("job_description")
        if description:
            with st.expander("Lihat deskripsi lengkap"):
                st.markdown(description_to_markdown(description))


def _render_llm_usage(result: AgentResult, seconds: float | None) -> None:
    """Expander rincian setiap panggilan LLM: langkah, model, dan token."""
    usage = result.usage
    label = (
        f"Detail: {SOURCE_LABELS.get(result.source, result.source)} | "
        f"{usage.llm_calls} panggilan LLM | {_number(usage.total_tokens)} token"
    )
    if seconds is not None:
        label += f" | {seconds:.1f} detik"
    with st.expander(label):
        if usage.calls:
            st.dataframe(
                [
                    {
                        "Langkah": STEP_LABELS.get(call.step, call.step),
                        "Model": call.model,
                        "Input": call.input_tokens,
                        "Output": call.output_tokens,
                        "Total": call.total_tokens,
                    }
                    for call in usage.calls
                ],
                hide_index=True,
            )
        else:
            st.caption("Tidak ada panggilan LLM.")
        rag_searches = sum(part.source == "rag" for part in result.parts or [result])
        if rag_searches:
            embedding_model = os.getenv("LLM_EMBEDDING_MODEL", "-")
            st.caption(
                f"Ditambah {rag_searches} embedding pertanyaan ({embedding_model}) untuk "
                "pencarian di Qdrant. Jumlah tokennya tidak dilaporkan oleh API embedding."
            )


# --- Helper tanpa Streamlit (bisa dites langsung) ----------------------------


def prepare_display(result: AgentResult) -> Display:
    """Tentukan teks dan kartu untuk satu jawaban.

    - RAG: kartu = ``result.rows`` (lowongan yang dikutip), catatan = alasan per ID
      dari jawaban. Jika ada lowongan yang barisnya tidak sesuai format, teks
      jawaban ditampilkan utuh dan kartu tanpa catatan.
    - SQL dan CV: jika jawaban memuat baris ``- [ID:x] ...``, detail lowongannya diambil
      dari SQLite agar kartunya sama dengan RAG. Jika tidak (misalnya jawaban
      jumlah, rata-rata, atau skill gap), hanya teks dan tabel.
    - Gabungan (``parts`` terisi): aturan di atas diterapkan per bagian.
    - Lainnya: hanya teks.
    """
    if result.parts:
        return Display("", parts=[prepare_display(part) for part in result.parts])
    if not result.job_ids:
        return Display(result.answer)
    intro, notes = split_rag_answer(result.answer)

    if result.source == "rag":
        if all(job["id"] in notes for job in result.rows):
            return Display(intro, [(job, notes[job["id"]]) for job in result.rows])
        return Display(result.answer, [(job, None) for job in result.rows])

    allowed = set(result.job_ids)
    cited = [job_id for job_id in notes if job_id in allowed][:MAX_CARDS]
    if not cited:
        return Display(result.answer)
    try:
        jobs = get_job_repository().get_jobs_by_ids(cited)
    except Exception:  # kartu hanya pelengkap: jika gagal, jawaban teks tetap tampil
        logger.exception("Gagal mengambil detail lowongan untuk kartu SQL")
        return Display(result.answer)
    return Display(intro, [(job, notes.get(job["id"])) for job in jobs])


def cv_payload(cv: dict[str, Any] | None) -> dict[str, Any] | None:
    """Bagian ``cv`` body API dari state CV aktif; ``None`` jika belum ada CV.

    Profil hanya memuat ``CV_PROFILE_FIELDS``, sehingga nama, kontak, lokasi, riwayat kerja,
    dan pencapaian tidak keluar dari proses Streamlit. ``matches`` dikirim utuh: isinya data
    lowongan dan skor, dibutuhkan agent CV untuk jawaban lowongan teratas dan skill gap.
    """
    if not cv:
        return None
    return {
        "profile": cv["profile"].model_dump(mode="json", include=CV_PROFILE_FIELDS),
        "matches": cv["matches"].model_dump(mode="json"),
    }


def cv_banner(cv: dict[str, Any]) -> str:
    """Teks banner CV aktif: nama file serta posisi tujuan dan rentang pengalaman."""
    label = escape_markdown(profile_label(cv["profile"])) or "profil kosong"
    return CV_BANNER.format(file_name=escape_markdown(cv["file_name"]), label=label)


def split_rag_answer(answer: str) -> tuple[str, dict[int, str]]:
    """Pisahkan jawaban menjadi teks pembuka dan catatan per ID lowongan.

    Format baris dari prompt RAG/SQL: ``- [ID:123] Judul - Perusahaan (Lokasi): catatan``.
    Catatan adalah teks setelah ``"): "`` pertama. Untuk lokasi bertingkat
    seperti ``(Jakarta Raya (Hibrid))`` pemisahnya tetap ditemukan, karena ``)``
    pertama diikuti ``)``, bukan ``:``.

    Baris kosong di pembuka dipertahankan (beberapa baris kosong berurutan jadi satu): tanpa
    pemisah itu, paragraf setelah daftar skill jawaban CV ikut menjadi bagian butir terakhir
    saat dirender sebagai markdown.

    Returns:
        ``(pembuka, {id: catatan})``. Baris tanpa ``[ID:x]`` masuk ke pembuka.
    """
    intro_lines: list[str] = []
    reasons: dict[int, str] = {}
    for line in answer.splitlines():
        match = _ANSWER_JOB_LINE.match(line)
        if match is None:
            text = line.strip()
            if text or (intro_lines and intro_lines[-1]):
                intro_lines.append(text)
            continue
        rest = match.group(2)
        _, separator, reason = rest.partition("): ")
        reasons[int(match.group(1))] = (reason if separator else rest).strip(" -:")
    return "\n".join(intro_lines).strip(), reasons


def _number(value: int) -> str:
    """Format angka Indonesia: 2400 -> "2.400"."""
    return f"{value:,}".replace(",", ".")
