"""Halaman Asisten Karier: chat yang memanggil supervisor agent.

Komponen ini hanya mengurus tampilan dan riwayat chat. Pemilihan dan
penggabungan agent (SQL, RAG, CV, atau pesan umum) sepenuhnya dikerjakan
``run_supervisor``; UI tidak memanggil agent lain secara langsung. Jawaban
gabungan (``source="multi"``) ditampilkan per bagian dengan aturan yang sama
seperti jawaban satu agent.

CV dibaca dari ``st.session_state["cv"]`` yang diisi halaman Cocokkan CV; halaman ini tidak
punya upload sendiri. Jika ada, profil dan hasil pencocokan diteruskan ke ``run_supervisor``
(planner menerima profil ringkas, teks CV tidak pernah dikirim). Banner di atas chat
menunjukkan apakah CV dipakai dan menautkan ke Cocokkan CV.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from agents.rag_agent import format_salary
from agents.supervisor_agent import run_supervisor
from components.job_description import description_to_markdown, escape_markdown
from cv.ingest import CV_STATE_KEY
from models.agent import AgentResult
from repositories.job_repository import get_jobs_by_ids

logger = logging.getLogger(__name__)

# Prefix "assistant_" agar tidak bentrok dengan key session_state halaman lain.
MESSAGES_KEY = "assistant_messages"
PENDING_KEY = "assistant_pending_question"
CV_PAGE = "pages/cvmatch.page.py"
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
    "CV aktif: {file_name} · {summary}. Profil ringkas (posisi tujuan, pengalaman, "
    "pendidikan, skill) dipakai untuk menjawab; teks CV tidak dikirim."
)
BANNER_SKILLS = 4
DEPENDENCY_ERROR = (
    "Asisten belum bisa dijalankan karena konfigurasi LLM atau Qdrant belum lengkap. "
    "Periksa file .env."
)
SOURCE_LABELS = {
    "rag": "Pencarian lowongan (RAG)",
    "sql": "Statistik data (SQL)",
    "general": "Pesan umum",
    "multi": "Gabungan beberapa agent",
    "cv": "Cocokkan CV (tanpa LLM)",
}
STEP_LABELS = {
    "planner": "Planner: pecah pertanyaan",
    "sql_generate": "SQL: tulis query",
    "sql_retry": "SQL: perbaiki query",
    "sql_answer": "SQL: rangkum hasil",
    "rag_answer": "RAG: jawab dari kandidat",
}
# Label catatan di kartu: alasan relevansi (RAG) atau fakta dari query (SQL, misalnya gaji).
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


@dataclass(frozen=True)
class AssistantDependencies:
    """Dependency yang diteruskan ke ``run_supervisor``."""

    llm: Any
    vector_store: Any
    db_path: str


def load_dependencies() -> AssistantDependencies:
    """Ambil LLM, vector store, dan path database dari ``config``.

    Import sengaja di dalam fungsi, bukan di atas file: jika ``.env`` belum
    lengkap, hanya halaman ini yang menampilkan pesan error, halaman lain
    tetap jalan. Import Python di-cache di ``sys.modules``, jadi klien LLM dan
    koneksi Qdrant tetap dibuat sekali per proses walaupun Streamlit rerun.
    """
    from config.database import VECTOR_DB
    from config.llm import LLM_MODEL

    return AssistantDependencies(
        llm=LLM_MODEL,
        vector_store=VECTOR_DB,
        db_path=os.getenv("DB_SQLITE_PATH", "./data/database.sqlite"),
    )


def render_assistant_page() -> None:
    """Render halaman chat lengkap: riwayat, contoh pertanyaan, dan input."""
    st.markdown("## Asisten Karier")
    st.caption("Tanyakan lowongan atau statistik lowongan dari data kami.")

    try:
        deps = load_dependencies()
    except Exception as exc:  # error konfigurasi bisa KeyError, koneksi Qdrant, dll.
        logger.exception("Gagal memuat dependency asisten")
        st.error(DEPENDENCY_ERROR)
        st.caption(f"Detail: {exc!r}")
        return

    cv: dict[str, Any] | None = st.session_state.get(CV_STATE_KEY)
    _render_cv_banner(cv)

    messages: list[dict[str, Any]] = st.session_state.setdefault(MESSAGES_KEY, [])
    for message in messages:
        _render_message(message)

    question = st.chat_input("Tanya tentang lowongan...") or st.session_state.pop(PENDING_KEY, None)
    if question:
        _answer(question, deps, messages, cv)
    elif not messages:
        _render_examples(CV_EXAMPLE_QUESTIONS if cv else EXAMPLE_QUESTIONS)

    # Dirender terakhir agar langsung muncul setelah jawaban pertama.
    if messages:
        st.button("Mulai percakapan baru", on_click=_clear_messages, key="assistant_clear")


# --- Alur tanya-jawab --------------------------------------------------------


def _answer(
    question: str,
    deps: AssistantDependencies,
    messages: list[dict[str, Any]],
    cv: dict[str, Any] | None,
) -> None:
    """Kirim pertanyaan ke supervisor, tampilkan jawaban, lalu simpan ke riwayat.

    ``history`` diambil sebelum pertanyaan baru ditambahkan, sesuai kontrak
    ``run_supervisor`` (riwayat tanpa pertanyaan saat ini). ``AgentResult``
    dan durasi ikut disimpan agar kartu dan detail LLM tetap tampil saat rerun.
    Profil dan hasil pencocokan CV aktif (jika ada) dibaca pada saat bertanya, jadi
    koreksi profil di Cocokkan CV langsung berlaku untuk pertanyaan berikutnya.
    """
    history = [{"role": message["role"], "content": message["content"]} for message in messages]
    user_message = {"role": "user", "content": question}
    messages.append(user_message)
    _render_message(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            start = time.perf_counter()
            result = run_supervisor(
                question,
                llm=deps.llm,
                vector_store=deps.vector_store,
                db_path=deps.db_path,
                history=history,
                profile=cv["profile"] if cv else None,
                matches=cv["matches"] if cv else None,
            )
            seconds = time.perf_counter() - start
            display = prepare_display(result, deps.db_path)
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


def _ask(question: str) -> None:
    """Callback tombol contoh: pertanyaan diproses pada rerun berikutnya."""
    st.session_state[PENDING_KEY] = question


def _clear_messages() -> None:
    """Callback tombol percakapan baru."""
    st.session_state[MESSAGES_KEY] = []


# --- Tampilan ----------------------------------------------------------------


def _render_cv_banner(cv: dict[str, Any] | None) -> None:
    """Status CV di atas chat dan tombol ke Cocokkan CV (0 LLM).

    ``st.switch_page`` dipanggil saat tombol diklik, bukan ``st.page_link``, supaya alurnya
    sama dengan tombol halaman lain dan bisa dites dengan AppTest.
    """
    with st.container(horizontal=True, vertical_alignment="center", wrap=False):
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
        result: AgentResult | None = message.get("result")
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


def cv_banner(cv: dict[str, Any]) -> str:
    """Teks banner CV aktif: nama file, posisi tujuan, pengalaman, dan beberapa skill."""
    profile = cv["profile"]
    parts = [escape_markdown(profile.target_role)] if profile.target_role else []
    if profile.experience_months is not None:
        parts.append(profile.experience_label)
    skills = [escape_markdown(skill) for skill in profile.skills[:BANNER_SKILLS]]
    if skills:
        more = len(profile.skills) - BANNER_SKILLS
        parts.append(", ".join(skills) + (f" +{more}" if more > 0 else ""))
    return CV_BANNER.format(
        file_name=escape_markdown(cv["file_name"]), summary=" · ".join(parts) or "profil kosong"
    )


def prepare_display(result: AgentResult, db_path: str) -> Display:
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
        return Display("", parts=[prepare_display(part, db_path) for part in result.parts])
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
        jobs = get_jobs_by_ids(db_path, cited)
    except Exception:  # kartu hanya pelengkap: jika gagal, jawaban teks tetap tampil
        logger.exception("Gagal mengambil detail lowongan untuk kartu SQL")
        return Display(result.answer)
    return Display(intro, [(job, notes.get(job["id"])) for job in jobs])


def split_rag_answer(answer: str) -> tuple[str, dict[int, str]]:
    """Pisahkan jawaban menjadi teks pembuka dan catatan per ID lowongan.

    Format baris dari prompt RAG/SQL: ``- [ID:123] Judul - Perusahaan (Lokasi): catatan``.
    Catatan adalah teks setelah ``"): "`` pertama. Untuk lokasi bertingkat
    seperti ``(Jakarta Raya (Hibrid))`` pemisahnya tetap ditemukan, karena ``)``
    pertama diikuti ``)``, bukan ``:``.

    Returns:
        ``(pembuka, {id: catatan})``. Baris tanpa ``[ID:x]`` masuk ke pembuka, dipisah baris
        kosong agar Markdown merendernya sebagai paragraf terpisah (judul bagian jawaban
        gabungan tidak menempel pada kalimat pembuka).
    """
    intro_lines: list[str] = []
    reasons: dict[int, str] = {}
    for line in answer.splitlines():
        match = _ANSWER_JOB_LINE.match(line)
        if match is None:
            if line.strip():
                intro_lines.append(line.strip())
            continue
        rest = match.group(2)
        _, separator, reason = rest.partition("): ")
        reasons[int(match.group(1))] = (reason if separator else rest).strip(" -:")
    return "\n\n".join(intro_lines), reasons


def _number(value: int) -> str:
    """Format angka Indonesia: 2400 -> "2.400"."""
    return f"{value:,}".replace(",", ".")
