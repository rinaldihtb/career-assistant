"""Supervisor StateGraph yang memecah pertanyaan ke worker RAG, SQL, dan CV.

Input graph: ``messages`` (riwayat lalu pertanyaan terakhir), serta ``profile`` dan
``matches`` jika CV pengguna aktif. Tanpa CV, prompt dan skema planner sama dengan versi
tanpa fitur CV. Dengan CV, planner memakai ``CVPlan`` (worker tambahan ``cv_match``,
``cv_gap``, dan ``cv_role``) dan menerima profil ringkas dari ``format_profile``; nama, kontak,
lokasi, riwayat kerja, dan teks CV tidak dikirim ke LLM.
"""

import operator
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel

from src.agents.cv_agent import run_cv_agent
from src.agents.prompts import (
    SUPERVISOR_CHAT_CV_NOTE,
    SUPERVISOR_CHAT_SYSTEM,
    SUPERVISOR_CHAT_USER,
    SUPERVISOR_CLARIFY_MESSAGE,
    SUPERVISOR_CLARIFY_SYSTEM,
    SUPERVISOR_DECLINED_NOTE,
    SUPERVISOR_GENERAL_CV_MESSAGE,
    SUPERVISOR_GENERAL_MESSAGE,
    SUPERVISOR_NO_HISTORY,
    SUPERVISOR_PLANNER_CV_AGENTS,
    SUPERVISOR_PLANNER_CV_RULES,
    SUPERVISOR_PLANNER_CV_USER,
    SUPERVISOR_PLANNER_SYSTEM,
    SUPERVISOR_PLANNER_USER,
    SUPERVISOR_REFUSE_MESSAGE,
    SUPERVISOR_SKIPPED_NOTE,
    SUPERVISOR_UNSUPPORTED_MESSAGE,
)
from src.agents.rag_agent import RagAgent, RetrievalResult
from src.agents.sql_agent import SQLQueryResult, finalize_answer
from src.cv.match import MatchResult, match_role
from src.cv.profile import CVProfile
from src.models.agent import AgentResult, LLMUsage
from src.repositories.job_repository import JobRepository, sqlite_path

MAX_SUBTASKS = 3
AgentName = Literal["sql", "rag"]
CVAgentName = Literal["sql", "rag", "cv_match", "cv_gap", "cv_role"]
Response = Literal["answer", "clarify", "refuse", "out_of_scope", "unsupported"]
# Respons planner saat CV aktif: tanpa "clarify", karena profil sudah memberi konteks. Dengan
# "clarify" tersedia, aturan clarify prompt tanpa CV membuat sapaan (C5-03) dan "Ada info
# kerjaan?" (C8-01) dijawab clarify di 4 dari 4 run walau aturan CV melarangnya.
CVResponse = Literal["answer", "refuse", "out_of_scope", "unsupported"]
_CITATION = re.compile(r"\[ID:\s*(\d+)\]", re.IGNORECASE)
_UNSAFE_CHAT = re.compile(r"\[ID:|Rp\s?\d", re.IGNORECASE)
MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_CHARS = 600
AGENT_NODES = {
    "sql": "sql_agent",
    "rag": "rag_agent",
    "cv_match": "cv_agent",
    "cv_gap": "cv_agent",
    "cv_role": "cv_agent",
}
# CV fiktif dengan skill terbanyak di set eval punya 15 skill; profil ringkas tetap < 150 token.
MAX_PROFILE_SKILLS = 15
# Tingkat jabatan yang dibuang dari posisi tujuan. Senioritas selalu dibuang; tingkat lain
# hanya jika sisanya minimal dua kata, agar "Account Executive" dan "HR Officer" tetap utuh.
SENIORITY_WORDS = frozenset({"senior", "junior", "sr", "jr"})
LEVEL_WORDS = frozenset({"executive", "staff", "officer"})
# Pemisah beberapa posisi dalam satu posisi tujuan ("A dan B", "A / B", "A, B"); "&" bukan
# pemisah karena "Sales & Marketing" satu posisi.
_ROLE_SEPARATOR = re.compile(r"\s+(?:dan|atau|and|or)\s+|\s*[/,;|]\s*", re.IGNORECASE)
# Syarat gaji berangka ("range gaji 5 juta sampai 8 juta"). RAG tidak bisa memfilter gaji, tetapi
# planner mengirim T3-01 ke rag di 6 dari 7 run walau prompt mewajibkan sql.
_SALARY_FILTER = re.compile(r"\bgaji\b[^.?!\n]*\d", re.IGNORECASE)
# Frasa yang merujuk pengguna, dengan kata depan opsional di depannya.
_SELF_REFERENCE = re.compile(
    r"\b(?:(?P<sesuai>sesuai)\s+(?:dengan\s+)?|(?P<prep>dengan|untuk|buat|bagi)\s+)?"
    r"(?P<ref>cv[\s-]?(?:ku|saya|aku)"
    r"|profil[\s-]?(?:ku|saya|aku)"
    r"|posisi\s+(?:yang\s+)?(?:ku|saya\s+|aku\s+)tuju"
    r"|(?:posisi\s+)?(?:tujuan|incaran)[\s-]?(?:ku|saya|aku)"
    r"|(?:untuk|buat|bagi)[\s-]?(?:ku|saya|aku))\b",
    re.IGNORECASE,
)
_PREPOSITION_REF = re.compile(r"(?:untuk|buat|bagi)", re.IGNORECASE)
# Kata yang meminta hasil Cocokkan CV. Tanpa kata ini, cv_match di samping rag/sql adalah
# tambahan yang tidak diminta ("carikan saya lowongan akuntan" -> rag + cv_match).
_MATCH_REQUEST = re.compile(r"\b(?:cocok|pas|sesuai|rekomendasi\w*)\b", re.IGNORECASE)
# Teks setelah "menjadi", "ke bidang", dan sejenisnya adalah posisi tujuan. Planner menulis kalimat
# utuh sebagai pertanyaan cv_role walau diminta nama posisi (C10-02, C10-03, C13-01).
_ROLE_TARGET = re.compile(
    r"\b(?:menjadi|jadi|sebagai|ke\s+bidang|ke\s+posisi|beralih\s+ke|pindah\s+ke)\s+(?P<role>.+)$",
    re.IGNORECASE,
)
# Pindah karier tanpa posisi tujuan; planner menjawabnya cv_match di C13-01 (1 dari 4 run).
_CAREER_SWITCH = re.compile(
    r"\b(?:switch|pindah|ganti|beralih|banting\s+setir)\b.*\b(?:karir|karier|profesi|bidang)\b",
    re.IGNORECASE,
)
# Pernyataan pindah ke posisi atau bidang yang disebut ("switch ke finance"). Planner membalasnya
# out_of_scope di C10-05 pada 7 dari 7 run, juga dengan aturan prompt yang memasukkannya ke cv_role.
_CAREER_MOVE = re.compile(
    r"\b(?:switch|pindah|beralih|ganti|banting\s+setir|upgrade|naik)"
    r"(?:\s+(?:karir|karier|profesi|bidang|posisi|jabatan))?"
    r"\s+(?:ke|jadi|menjadi)\s+(?:bidang\s+|posisi\s+)?(?P<role>[^?.!,;\n]+)",
    re.IGNORECASE,
)
# Isi cv_role yang bukan nama posisi, misalnya "bidang lain" untuk "cara switch karir" (C13-01).
_GENERIC_ROLE_WORDS = frozenset(
    {"lain", "baru", "bidang", "posisi", "karier", "karir", "profesi", "kerja", "pekerjaan"}
)


class SubTask(BaseModel):
    """Satu pekerjaan yang diberikan planner kepada satu worker."""

    agent: AgentName
    question: str


class Plan(BaseModel):
    """Output structured dari planner."""

    response: Response
    subtasks: list[SubTask]
    skipped: list[str]
    declined: list[str]


class CVSubTask(BaseModel):
    """Sub-tugas saat CV aktif: sama dengan ``SubTask`` ditambah worker CV."""

    agent: CVAgentName
    question: str


class CVPlan(BaseModel):
    """Output planner saat CV aktif.

    Skema terpisah dari ``Plan`` supaya skema yang dikirim tanpa CV tetap sama dan planner
    tanpa CV tidak bisa memilih worker CV. ``response`` tidak memuat "clarify"; structured
    output OpenAI menegakkan enum ini.
    """

    response: CVResponse
    subtasks: list[CVSubTask]
    skipped: list[str]
    declined: list[str]


class SupervisorState(TypedDict, total=False):
    """State bersama dari planner sampai jawaban akhir."""

    messages: Annotated[list[BaseMessage], add_messages]
    profile: CVProfile | None
    matches: MatchResult | None
    response: Response
    subtasks: list[dict[str, str]]
    skipped: list[str]
    declined: list[str]
    planner_usage: LLMUsage
    worker_index: int
    worker_question: str
    worker_agent: str
    collected: Annotated[list[tuple[int, AgentResult]], operator.add]
    subresults: list[AgentResult]
    result: AgentResult


FIXED_RESPONSES = {
    "clarify": SUPERVISOR_CLARIFY_MESSAGE,
    "refuse": SUPERVISOR_REFUSE_MESSAGE,
    "out_of_scope": SUPERVISOR_GENERAL_MESSAGE,
    "unsupported": SUPERVISOR_UNSUPPORTED_MESSAGE,
}


def _notes(skipped: list[str], declined: list[str]) -> str:
    """Buat catatan untuk bagian pertanyaan yang tidak dikerjakan."""
    notes = []
    if declined:
        notes.append(SUPERVISOR_DECLINED_NOTE)
    if skipped:
        notes.append(SUPERVISOR_SKIPPED_NOTE + "\n" + "\n".join(f"- {item}" for item in skipped))
    return "\n\n".join(notes)


def check_citations(answer: str, candidate_ids: list[int]) -> tuple[list[int], list[int]]:
    """Pisahkan ID yang dikutip LLM menjadi kandidat valid dan ID di luar kandidat."""
    candidates = set(candidate_ids)
    cited = list(dict.fromkeys(int(match) for match in _CITATION.findall(answer)))
    return (
        [job_id for job_id in cited if job_id in candidates],
        [job_id for job_id in cited if job_id not in candidates],
    )


def usage_from_messages(messages: list[BaseMessage], step: str) -> LLMUsage:
    """Catat token dari setiap respons LLM dalam satu worker."""
    usage = LLMUsage()
    for message in messages:
        if isinstance(message, AIMessage):
            usage.add(message, step=step)
    return usage


def _chat_context(messages: Sequence[BaseMessage]) -> tuple[str, str, str]:
    """Pisahkan pertanyaan terakhir dari riwayat untuk prompt percakapan umum."""
    items = list(messages)[-MAX_HISTORY_MESSAGES:]
    last_user_index = next(
        (index for index in range(len(items) - 1, -1, -1) if _message_role(items[index]) == "user"),
        None,
    )
    if last_user_index is None:
        return "", SUPERVISOR_NO_HISTORY, SUPERVISOR_NO_HISTORY

    question = _message_content(items[last_user_index])
    history_items = items[:last_user_index]
    history_lines = [
        f"{'Pengguna' if _message_role(item) == 'user' else 'Asisten'}: {_message_content(item)}"
        for item in history_items
        if _message_content(item)
    ]
    questions = [
        _message_content(item)
        for item in history_items
        if _message_role(item) == "user" and _message_content(item)
    ]
    history = "\n".join(history_lines) or SUPERVISOR_NO_HISTORY
    previous_questions = (
        "\n".join(f"{index}. {item}" for index, item in enumerate(questions, 1))
        or SUPERVISOR_NO_HISTORY
    )
    return question, history, previous_questions


def _message_role(message: BaseMessage | Mapping[str, object]) -> Literal["user", "assistant"]:
    """Normalisasi peran pesan dari objek LangChain atau dict riwayat UI."""
    if isinstance(message, Mapping):
        return "user" if message.get("role") in {"user", "human"} else "assistant"
    return "user" if message.type == "human" else "assistant"


def _message_content(message: BaseMessage | Mapping[str, object]) -> str:
    """Ambil konten pesan, rapikan spasi, dan batasi panjang prompt."""
    content = message.get("content", "") if isinstance(message, Mapping) else message.content
    text = " ".join(str(content).split())
    return f"{text[:MAX_HISTORY_CHARS]}..." if len(text) > MAX_HISTORY_CHARS else text


def planner_messages(
    question: str, history: str, profile: CVProfile | None
) -> list[SystemMessage | HumanMessage]:
    """Pesan planner. Tanpa profil, slot CV di prompt sistem diisi string kosong.

    Dengan profil, prompt sistem mendapat worker ``cv_match``/``cv_gap`` beserta aturannya,
    dan pesan pengguna diawali profil ringkas di tag ``<profil>``.
    """
    if profile is None:
        system = SUPERVISOR_PLANNER_SYSTEM.format(
            max_subtasks=MAX_SUBTASKS, cv_agents="", cv_rules=""
        )
        user = SUPERVISOR_PLANNER_USER.format(history=history, question=question)
    else:
        system = SUPERVISOR_PLANNER_SYSTEM.format(
            max_subtasks=MAX_SUBTASKS,
            cv_agents=SUPERVISOR_PLANNER_CV_AGENTS,
            cv_rules=SUPERVISOR_PLANNER_CV_RULES,
        )
        user = SUPERVISOR_PLANNER_CV_USER.format(
            profile=format_profile(profile), history=history, question=question
        )
    return [SystemMessage(system), HumanMessage(user)]


def format_profile(profile: CVProfile) -> str:
    """Profil ringkas untuk planner: posisi tujuan inti, pengalaman, pendidikan, skill.

    Planner hanya butuh bahan untuk menulis pertanyaan mandiri ("lowongan <posisi> di <kota>"),
    jadi nama, kontak, lokasi, perusahaan, dan riwayat kerja tidak dikirim. Skill dibatasi
    ``MAX_PROFILE_SKILLS`` dengan urutan profil; baris kosong dilewati. Posisi tujuan dikirim
    tanpa tingkat jabatan (``target_roles``): teks mentah "Sales & Marketing Executive" disalin
    planner ke pertanyaan SQL/RAG walau aturan prompt melarang (C3-04, C4-03).
    """
    education = " ".join(p for p in (profile.education_level, profile.education_major) if p)
    skills = profile.skills[:MAX_PROFILE_SKILLS]
    roles = " atau ".join(target_roles(profile.target_role))
    lines = [
        f"Posisi tujuan: {roles}" if roles else "",
        f"Pengalaman: {profile.experience_label}",
        f"Pendidikan: {education}" if education else "",
        f"Skill: {', '.join(skills)}" if skills else "",
    ]
    return "\n".join(line for line in lines if line)


def core_role(role: str) -> str:
    """Satu posisi tanpa tingkat jabatan: "Senior Teknisi Listrik" -> "Teknisi Listrik".

    ``SENIORITY_WORDS`` selalu dibuang; ``LEVEL_WORDS`` hanya jika sisanya minimal dua kata
    ("Sales & Marketing Executive" -> "Sales & Marketing", "Account Executive" tetap). Tanda
    baca tidak dihitung sebagai kata.
    """
    words = role.split()
    without_seniority = [w for w in words if w.lower().strip(".") not in SENIORITY_WORDS] or words
    core = [w for w in without_seniority if w.lower().strip(".") not in LEVEL_WORDS]
    named = [w for w in core if any(ch.isalnum() for ch in w)]
    return " ".join(core if len(named) >= 2 else without_seniority)


def target_roles(target_role: str | None) -> list[str]:
    """Posisi tujuan dipecah per posisi lalu diambil intinya, tanpa duplikat, urutan tetap.

    Profil bisa berisi dua posisi dalam satu teks ("Fullstack Developer dan Data Analyst");
    sebagai satu pola judul hasilnya 0 lowongan, sedangkan '%data analyst%' 30.
    """
    roles: list[str] = []
    for part in _ROLE_SEPARATOR.split(target_role or ""):
        role = core_role(part.strip())
        if role and role.lower() not in {r.lower() for r in roles}:
            roles.append(role)
    return roles


def profile_role(profile: CVProfile) -> str:
    """Posisi tujuan inti untuk pertanyaan mandiri SQL/RAG; tanpa posisi, 1-2 skill pertama.

    Posisi digabung "atau", sama dengan baris "Posisi tujuan" di ``format_profile``; string
    kosong jika profil tidak punya posisi maupun skill.
    """
    return " atau ".join(target_roles(profile.target_role) or profile.skills[:2])


def role_from_question(question: str) -> str:
    """Nama posisi dari pertanyaan sub-tugas ``cv_role``.

    "Skill yang perlu kupelajari untuk menjadi Product Manager" -> "Product Manager"; teks tanpa
    penanda posisi dipakai utuh. Spasi dirapikan dan tanda baca di ujung dibuang.
    """
    text = " ".join(question.split())
    match = _ROLE_TARGET.search(text)
    return (match.group("role") if match else text).strip(" .,;:!?")


def is_new_role(role: str, profile: CVProfile) -> bool:
    """``True`` jika ``role`` posisi yang bisa dicari dan berbeda dari posisi tujuan di CV.

    Posisi yang hanya berisi kata umum ("bidang lain") atau sama dengan posisi tujuan profil
    (tanpa tingkat jabatan) bukan permintaan posisi lain: C13-01 "cara switch karir" diberi
    ``cv_role`` "Data Analyst" untuk CV Data Analyst di 3 dari 4 run.
    """
    words = role.lower().split()
    if all(word in _GENERIC_ROLE_WORDS for word in words):
        return False
    own = {own_role.lower() for own_role in target_roles(profile.target_role)}
    return core_role(role).lower() not in own


def career_move_role(question: str, profile: CVProfile) -> str:
    """Posisi tujuan dari pernyataan pindah karier: "switch ke finance" -> "finance".

    String kosong jika pertanyaan tidak memuat kata kerja pindah yang diikuti tujuan, atau
    tujuannya bukan posisi baru (``is_new_role``), misalnya "pindah ke bidang lain".
    """
    match = _CAREER_MOVE.search(" ".join(question.split()))
    if match is None:
        return ""
    role = match.group("role").strip(" .,;:!?")
    return role if is_new_role(role, profile) else ""


def replace_self_references(question: str, role: str) -> str:
    """Ganti frasa yang merujuk pengguna ("CV-ku", "untukku", "posisi yang kutuju") dengan ``role``.

    "cocok untukku" -> "cocok untuk <role>", "sesuai CV-ku" -> "sesuai untuk <role>",
    "posisi incaranku" -> "<role>". Spasi berlebih dirapikan; ``role`` kosong membuat
    pertanyaan dikembalikan apa adanya.
    """
    if not role:
        return question

    def replacement(match: re.Match[str]) -> str:
        if match.group("sesuai"):
            return f"sesuai untuk {role}"
        if match.group("prep") or _PREPOSITION_REF.match(match.group("ref")):
            return f"untuk {role}"
        return role

    return " ".join(_SELF_REFERENCE.sub(replacement, question).split())


class SupervisorAgent:
    """Planner memilih satu atau beberapa worker, lalu graph menggabungkan jawabannya."""

    def __init__(
        self,
        llm: BaseChatModel,
        job_repository: JobRepository,
        rag_agent: CompiledStateGraph,
        sql_agent: CompiledStateGraph,
    ) -> None:
        self.job_repository = job_repository
        self.rag_agent = rag_agent
        self.sql_agent = sql_agent
        self.llm = llm
        self.graph = self.create_supervisor_agent()

    def planner(self, state: SupervisorState) -> dict:
        """Ubah pesan pengguna terakhir menjadi satu atau beberapa sub-tugas worker.

        Pertanyaan terakhir dan riwayat berlabel Pengguna/Asisten diambil dari
        ``messages`` (``_chat_context``) dan dikirim dalam satu pesan pengguna
        (``planner_messages``); dengan ``profile`` skemanya ``CVPlan``. Sub-tugas di atas
        ``MAX_SUBTASKS`` masuk ``skipped`` agar pengguna diberi tahu bagian yang belum
        dijawab. Dengan CV, balasan out_of_scope untuk pernyataan pindah ke posisi yang disebut
        diganti satu sub-tugas ``cv_role`` (``career_move_role``).
        """
        question, history, _ = _chat_context(state.get("messages", []))
        profile = state.get("profile")
        messages = planner_messages(question, history, profile)
        schema = Plan if profile is None else CVPlan
        fallback: dict = {"response": "clarify", "subtasks": [], "skipped": [], "declined": []}
        try:
            output = cast(
                dict[str, object],
                self.llm.with_structured_output(schema, include_raw=True).invoke(messages),
            )
        except Exception:  # noqa: BLE001 - provider/parsing error beragam.
            return {**fallback, "planner_usage": LLMUsage()}

        usage = LLMUsage()
        usage.add(output.get("raw"), step="planner")
        parsed_plan = output.get("parsed")
        if not isinstance(parsed_plan, schema):
            return {**fallback, "planner_usage": usage}

        subtasks = [
            {"agent": task.agent, "question": task.question.strip()}
            for task in parsed_plan.subtasks
            if task.question.strip()
        ]
        subtasks = self.align_subtasks(question, subtasks, profile)
        response: Response = parsed_plan.response
        if response == "answer" and not subtasks:
            # Tanpa CV planner bisa meminta klarifikasi; dengan CV, skema tanpa "clarify" dan
            # sub-tugas yang habis dibuang penjaga dibalas lewat out_of_scope (C13-01).
            response = "clarify" if profile is None else "out_of_scope"
        if profile is not None and response == "out_of_scope" and not subtasks:
            role = career_move_role(question, profile)
            if role:
                response, subtasks = "answer", [{"agent": "cv_role", "question": role}]
        overflow = [task["question"] for task in subtasks[MAX_SUBTASKS:]]
        return {
            "response": response,
            "subtasks": subtasks[:MAX_SUBTASKS],
            "skipped": [*parsed_plan.skipped, *overflow],
            "declined": parsed_plan.declined,
            "planner_usage": usage,
        }

    def align_subtasks(
        self, question: str, subtasks: list[dict[str, str]], profile: CVProfile | None
    ) -> list[dict[str, str]]:
        """Betulkan rute planner yang terukur salah walau dilarang prompt, tanpa LLM.

        - Sub-tugas ``rag`` dengan syarat gaji berangka dialihkan ke ``sql`` (T3-01).
        - CV aktif, pengguna merujuk dirinya dan menyebut kota, tetapi planner hanya memberi
          ``cv_match`` yang tidak bisa memfilter apa pun (C3-01 dan C3-03, 4 dari 4 run):
          ``cv_match`` diganti ``rag`` dengan pertanyaan pengguna yang rujukan dirinya diganti
          posisi tujuan, sehingga kota dan pengecualian kota tetap utuh. Jika lokasi tidak bisa
          dibaca dari database, rute planner dipakai apa adanya.
        - ``cv_match`` di samping ``rag``/``sql`` dibuang jika pertanyaan tidak memuat kata
          permintaan cocok (``_MATCH_REQUEST``): "carikan saya lowongan akuntan" dijawab daftar
          akuntan ditambah lowongan posisi CV yang tidak diminta (C11-01).
        - ``cv_match`` di samping ``rag`` yang sudah memuat posisi tujuan profil dan lokasi dibuang:
          ``rag`` itu sudah jawaban personal dengan kota (C3-03, 3 dari 3 run).
        - Sub-tugas ``cv_role`` dirapikan oleh ``align_role_subtasks``. Pertanyaan pindah karier
          tanpa ``cv_role`` yang hanya dijawab ``cv_match``/``cv_gap`` dikosongkan, sehingga
          planner membalas lewat out_of_scope yang menanyakan bidang tujuan (C13-01).
        """
        aligned = [
            {**task, "agent": "sql"}
            if task["agent"] == "rag" and _SALARY_FILTER.search(task["question"])
            else task
            for task in subtasks
        ]
        if profile is not None:
            aligned = self.align_role_subtasks(aligned, profile)
        agents = {task["agent"] for task in aligned}
        if "cv_match" in agents and agents & {"sql", "rag"} and not _MATCH_REQUEST.search(question):
            aligned = [task for task in aligned if task["agent"] != "cv_match"]
            agents.discard("cv_match")
        if (
            profile is not None
            and {"cv_match", "rag"} <= agents
            and self._has_personal_rag(aligned, profile)
        ):
            aligned = [task for task in aligned if task["agent"] != "cv_match"]
            agents.discard("cv_match")
        if (
            profile is not None
            and agents
            and agents <= {"cv_match", "cv_gap"}
            and _CAREER_SWITCH.search(question)
        ):
            return []
        if profile is None or "cv_match" not in agents or agents & {"sql", "rag"}:
            return aligned
        if not _SELF_REFERENCE.search(question):
            return aligned
        try:
            places = RagAgent(self.llm, self.job_repository).find_places(question)
        except Exception:  # noqa: BLE001 - deteksi lokasi opsional; rute planner tetap dipakai
            return aligned
        if not places:
            return aligned
        rag = {"agent": "rag", "question": replace_self_references(question, profile_role(profile))}
        return [rag if task["agent"] == "cv_match" else task for task in aligned]

    def align_role_subtasks(
        self, subtasks: list[dict[str, str]], profile: CVProfile
    ) -> list[dict[str, str]]:
        """Rapikan sub-tugas ``cv_role`` dan buang sub-tugas yang mengulang isinya, tanpa LLM.

        - Pertanyaan ``cv_role`` diganti nama posisinya (``role_from_question``); ``cv_role``
          yang bukan posisi baru (``is_new_role``) dibuang.
        - Jika ada ``cv_role``, ``cv_match`` dan ``cv_gap`` dibuang karena dihitung untuk posisi
          lama di CV, dan ``rag`` yang menyebut posisi yang sama tanpa kota dibuang karena
          lowongan teratasnya sudah ada di jawaban ``cv_role``. Planner menambah keduanya
          di C10-02, C10-03, dan C10-04 pada 11 dari 11 run.
        """
        roles: list[str] = []
        kept: list[dict[str, str]] = []
        for task in subtasks:
            if task["agent"] != "cv_role":
                kept.append(task)
                continue
            role = role_from_question(task["question"])
            if role and is_new_role(role, profile):
                roles.append(role.lower())
                kept.append({**task, "question": role})
        if not roles:
            return kept
        return [
            task
            for task in kept
            if task["agent"] not in ("cv_match", "cv_gap")
            and not (
                task["agent"] == "rag"
                and any(role in task["question"].lower() for role in roles)
                and not self._mentions_place(task["question"])
            )
        ]

    def _has_personal_rag(self, subtasks: list[dict[str, str]], profile: CVProfile) -> bool:
        """``True`` jika ada sub-tugas ``rag`` yang memuat posisi tujuan profil dan lokasi."""
        roles = [role.lower() for role in target_roles(profile.target_role)]
        return any(
            task["agent"] == "rag"
            and any(role in task["question"].lower() for role in roles)
            and self._mentions_place(task["question"], default=False)
            for task in subtasks
        )

    def _mentions_place(self, question: str, *, default: bool = True) -> bool:
        """``True`` jika pertanyaan menyebut lokasi dari database; ``default`` jika gagal dibaca."""
        try:
            return bool(RagAgent(self.llm, self.job_repository).find_places(question))
        except Exception:  # noqa: BLE001 - lokasi opsional; pemanggil memilih nilai aman
            return default

    def dispatch(self, state: SupervisorState) -> list[Send] | str:
        """Kirim setiap sub-tugas ke worker yang sesuai, atau langsung merespons."""
        response: Response = state.get("response", "out_of_scope")
        if response != "answer":
            return "respond"

        return [
            Send(
                AGENT_NODES[task["agent"]],
                {
                    "worker_index": index,
                    "worker_question": task["question"],
                    "worker_agent": task["agent"],
                    "profile": state.get("profile"),
                    "matches": state.get("matches"),
                },
            )
            for index, task in enumerate(state.get("subtasks", []))
        ]

    def run_rag_agent(self, state: SupervisorState) -> dict[str, list[tuple[int, AgentResult]]]:
        """Jalankan worker RAG untuk satu sub-tugas."""
        index = state.get("worker_index")
        question = state.get("worker_question")
        if index is None or not question:
            return {"collected": []}
        result = self.rag_agent.invoke({"messages": [HumanMessage(content=question)]})
        answer = result["messages"][-1].content
        usage = usage_from_messages(result["messages"], step="rag")
        retrieval = next(
            (
                message.artifact
                for message in reversed(result["messages"])
                if isinstance(message, ToolMessage)
                and message.name == "retrieve_jobs"
                and isinstance(message.artifact, RetrievalResult)
            ),
            None,
        )
        if retrieval is None:
            return {
                "collected": [(index, AgentResult(answer=str(answer), source="rag", usage=usage))]
            }

        valid_ids, invalid_ids = check_citations(str(answer), [job["id"] for job in retrieval.jobs])
        jobs_by_id = {job["id"]: job for job in retrieval.jobs}
        return {
            "collected": [
                (
                    index,
                    AgentResult(
                        answer=str(answer),
                        source="rag",
                        job_ids=valid_ids,
                        rows=[jobs_by_id[job_id] for job_id in valid_ids],
                        usage=usage,
                        error=(
                            f"ID di luar kandidat dibuang: {invalid_ids}" if invalid_ids else None
                        ),
                    ),
                )
            ]
        }

    def run_sql_agent(self, state: SupervisorState) -> dict[str, list[tuple[int, AgentResult]]]:
        """Jalankan worker SQL untuk satu sub-tugas."""
        index = state.get("worker_index")
        question = state.get("worker_question")
        if index is None or not question:
            return {"collected": []}
        result = self.sql_agent.invoke({"messages": [HumanMessage(content=question)]})
        answer = result["messages"][-1].content
        usage = usage_from_messages(result["messages"], step="sql")
        sql_data = next(
            (
                message.artifact
                for message in reversed(result["messages"])
                if isinstance(message, ToolMessage)
                and message.name == "query_jobs"
                and isinstance(message.artifact, SQLQueryResult)
            ),
            None,
        )
        if sql_data is not None:
            sql_result = AgentResult(
                answer=finalize_answer(str(answer), sql_data.rows),
                source="sql",
                job_ids=sql_data.job_ids,
                sql=sql_data.sql,
                rows=sql_data.rows,
                usage=usage.merged(sql_data.usage),
                error=sql_data.error,
            )
        else:
            sql_result = AgentResult(answer=str(answer), source="sql", usage=usage)
        return {"collected": [(index, sql_result)]}

    def run_cv_agent(self, state: SupervisorState) -> dict[str, list[tuple[int, AgentResult]]]:
        """Jalankan worker CV untuk satu sub-tugas, tanpa LLM.

        ``cv_match``/``cv_gap`` memakai hasil Cocokkan CV di state; ``cv_role`` mencocokkan
        ulang untuk posisi di pertanyaan sub-tugas (``match_role``, 1 embedding).
        """
        index = state.get("worker_index")
        kind = state.get("worker_agent")
        if index is None or kind not in ("cv_match", "cv_gap", "cv_role"):
            return {"collected": []}
        profile = state.get("profile")
        if kind == "cv_role":
            role = state.get("worker_question", "")
            result = run_cv_agent(
                kind=kind, profile=profile, matches=self.match_role(role, profile), role=role
            )
        else:
            result = run_cv_agent(kind=kind, profile=profile, matches=state.get("matches"))
        return {"collected": [(index, result)]}

    def match_role(self, role: str, profile: CVProfile | None) -> MatchResult:
        """Hasil pencocokan untuk posisi lain, dengan embedding milik vector store repository.

        Skill CV dari ``profile`` dipakai untuk skill cocok dan kurang. SQLite dibaca dari
        ``DB_SQLITE_PATH``, sama dengan halaman Cocokkan CV. Tidak pernah melempar exception.
        """
        store = self.job_repository.vector_db
        embeddings = getattr(store, "embeddings", None)
        if embeddings is None:
            return MatchResult(error="Vector store tidak punya model embedding")
        return match_role(
            role,
            profile.skills if profile is not None else [],
            embeddings=embeddings,
            vector_store=store,
            db_path=sqlite_path(),
        )

    def synthesize(self, state: SupervisorState) -> dict[str, AgentResult | list[AgentResult]]:
        """Gabungkan jawaban worker sesuai urutan pertanyaan."""
        subresults = [result for _, result in sorted(state.get("collected", []))]
        notes = _notes(state.get("skipped", []), state.get("declined", []))
        total_usage = state.get("planner_usage", LLMUsage())
        for result in subresults:
            total_usage = total_usage.merged(result.usage)

        if len(subresults) == 1 and not notes:
            result = subresults[0].model_copy(update={"usage": total_usage})
            return {"subresults": subresults, "result": result}

        tasks = state.get("subtasks", [])
        parts = [
            result.model_copy(update={"answer": f"**{task['question']}**\n{result.answer}"})
            if len(subresults) > 1
            else result
            for task, result in zip(tasks, subresults, strict=True)
        ]
        if notes:
            parts.append(AgentResult(answer=notes, source="general"))

        job_ids = list(dict.fromkeys(job_id for result in subresults for job_id in result.job_ids))
        return {
            "subresults": subresults,
            "result": AgentResult(
                answer="\n\n".join(part.answer for part in parts),
                source="multi",
                job_ids=job_ids,
                usage=total_usage,
                parts=parts,
            ),
        }

    def respond(self, state: SupervisorState) -> dict[str, AgentResult]:
        """Kirim respons tetap, atau balasan LLM yang luwes untuk percakapan umum.

        Saat CV aktif, balasan ``out_of_scope`` juga menyebut kemampuan berbasis CV
        (``SUPERVISOR_CHAT_CV_NOTE``) dan fallback-nya ``SUPERVISOR_GENERAL_CV_MESSAGE``.
        """
        response: Response = state.get("response", "out_of_scope")
        usage = state.get("planner_usage", LLMUsage())
        has_cv = state.get("profile") is not None
        if response == "out_of_scope":
            chat_usage = LLMUsage()
            answer = self._chat_reply(
                state,
                chat_usage,
                system_prompt=SUPERVISOR_CHAT_SYSTEM + (SUPERVISOR_CHAT_CV_NOTE if has_cv else ""),
                fallback=SUPERVISOR_GENERAL_CV_MESSAGE if has_cv else SUPERVISOR_GENERAL_MESSAGE,
            )
            usage = usage.merged(chat_usage)
        elif response == "clarify":
            chat_usage = LLMUsage()
            answer = self._chat_reply(
                state,
                chat_usage,
                system_prompt=SUPERVISOR_CLARIFY_SYSTEM,
                fallback=SUPERVISOR_CLARIFY_MESSAGE,
            )
            usage = usage.merged(chat_usage)
        else:
            answer = FIXED_RESPONSES[response]
        return {
            "result": AgentResult(
                answer=answer,
                source="general",
                usage=usage,
            )
        }

    def _chat_reply(
        self,
        state: SupervisorState,
        usage: LLMUsage,
        *,
        system_prompt: str,
        fallback: str,
    ) -> str:
        """Buat balasan LLM dengan riwayat percakapan dan fallback yang aman."""
        question, history, previous_questions = _chat_context(state.get("messages", []))
        if not question:
            return fallback

        try:
            response = self.llm.invoke(
                [
                    SystemMessage(system_prompt),
                    HumanMessage(
                        SUPERVISOR_CHAT_USER.format(
                            previous_questions=previous_questions,
                            history=history,
                            question=question,
                        )
                    ),
                ]
            )
        except Exception:  # noqa: BLE001 - error provider beragam.
            return fallback

        usage.add(response, step="chat")
        answer = str(getattr(response, "text", response.content)).strip()
        if not answer or _UNSAFE_CHAT.search(answer):
            return fallback
        return answer

    def create_supervisor_agent(self) -> CompiledStateGraph:
        """Bangun workflow supervisor.

        Alur utama::

            START → planner
                    ├→ respond → END
                    ├→ rag_agent ─┐
                    ├→ sql_agent ─┤
                    └→ cv_agent ──┴→ synthesize → END

        ``planner`` memilih apakah pertanyaan perlu dijawab dengan pesan tetap
        atau dipecah menjadi sub-tugas RAG, SQL, dan CV. ``dispatch`` dapat mengirim
        beberapa sub-tugas sekaligus melalui ``Send``. Hasil setiap worker
        dikumpulkan oleh ``synthesize`` sebelum graph selesai.
        """
        graph = StateGraph(SupervisorState)
        graph.add_node("planner", self.planner)
        graph.add_node("rag_agent", self.run_rag_agent)
        graph.add_node("sql_agent", self.run_sql_agent)
        graph.add_node("cv_agent", self.run_cv_agent)
        graph.add_node("synthesize", self.synthesize)
        graph.add_node("respond", self.respond)
        graph.add_edge(START, "planner")
        graph.add_conditional_edges("planner", self.dispatch)
        graph.add_edge("rag_agent", "synthesize")
        graph.add_edge("sql_agent", "synthesize")
        graph.add_edge("cv_agent", "synthesize")
        graph.add_edge("synthesize", END)
        graph.add_edge("respond", END)
        return graph.compile()


if __name__ == "__main__":
    from src.containers import get_supervisor_agent

    state = get_supervisor_agent().graph.invoke(
        {"messages": [HumanMessage(content="Lowongan data analyst jakarta?")]}
    )
    print(state["result"].answer)
