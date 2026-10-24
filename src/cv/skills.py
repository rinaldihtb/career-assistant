"""Pencocokan skill CV dengan deskripsi lowongan, tanpa LLM.

Dua sumber skill dipakai bersama:
- ``SKILL_VOCAB``: kosakata hard skill yang dikurasi dari 473 deskripsi lowongan; setiap
  istilah muncul di minimal 2 lowongan. Kosakata menentukan skill yang disebut lowongan,
  sehingga "skill kurang" bisa ditampilkan.
- Skill CV di luar kosakata (misalnya "YOLO", "SHAP", "LangChain") tetap dicari langsung di
  deskripsi, supaya tidak hilang hanya karena tidak ada di kosakata.

Porsi skill sebuah lowongan = skill yang disebut lowongan dan dimiliki CV, dibagi skill yang
disebut lowongan. Kode tidak membedakan skill wajib dan opsional ("nice to have"); keduanya
dihitung sama.

Soft skill sengaja tidak dihitung: "communication" muncul di 193 dari 473 lowongan, jadi
hanya menaikkan porsi CV yang kebetulan menuliskannya tanpa membedakan lowongan. Nama satu
huruf (R, C) tidak dipakai, dan "Go" hanya dikenali di daftar skill CV: di deskripsi, "go"
muncul sebagai "go beyond" dan "go-to-market".
"""

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass

# Nama tampilan -> alias. Alias dicocokkan tanpa membedakan huruf besar/kecil, dengan batas
# kata; spasi dan tanda hubung antarkata boleh berbeda ("power-bi", "Power  BI").
SKILL_VOCAB: dict[str, tuple[str, ...]] = {
    # Data dan AI
    "Python": ("python", "phyton"),
    "SQL": ("sql",),
    "Excel": ("excel", "microsoft excel", "ms excel"),
    "Google Sheets": ("google sheets", "google sheet", "gsheet", "gsheets"),
    "Power BI": ("power bi", "powerbi"),
    "Tableau": ("tableau",),
    "Looker": ("looker", "looker studio", "google data studio", "data studio"),
    "Machine Learning": ("machine learning", "ml"),
    "Deep Learning": ("deep learning",),
    "Computer Vision": ("computer vision",),
    "NLP": ("nlp", "natural language processing"),
    "LLM": ("llm", "llms", "large language model", "large language models"),
    "Generative AI": ("generative ai", "genai", "gen ai"),
    "RAG": ("rag", "retrieval-augmented generation", "retrieval augmented generation"),
    "TensorFlow": ("tensorflow",),
    "PyTorch": ("pytorch",),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "Pandas": ("pandas",),
    "NumPy": ("numpy",),
    "ETL": ("etl", "elt"),
    "Data Warehouse": ("data warehouse", "data warehousing"),
    "BigQuery": ("bigquery",),
    "Snowflake": ("snowflake",),
    "Redshift": ("redshift",),
    "Spark": ("spark", "pyspark", "apache spark"),
    "Airflow": ("airflow",),
    "Kafka": ("kafka",),
    "dbt": ("dbt",),
    "A/B Testing": ("a/b testing", "a/b test", "ab testing"),
    "Data Visualization": ("data visualization", "data visualisation", "visualisasi data"),
    "VBA": ("vba", "excel macros", "macro excel"),
    # Software
    "Java": ("java",),
    "JavaScript": ("javascript",),
    "TypeScript": ("typescript",),
    "PHP": ("php",),
    "Laravel": ("laravel",),
    "Golang": ("golang", "go-lang"),
    "C++": ("c++",),
    "C#": ("c#",),
    "React": ("react", "react.js", "reactjs", "react native"),
    "Vue.js": ("vue", "vue.js", "vuejs"),
    "Node.js": ("node.js", "nodejs"),
    "Angular": ("angular",),
    "HTML": ("html", "html5"),
    "CSS": ("css",),
    "Django": ("django",),
    "Spring Boot": ("spring boot", "springboot"),
    "Android": ("android",),
    "iOS": ("ios",),
    "REST API": ("api", "apis", "rest api", "restful"),
    "Git": ("git", "github", "gitlab"),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "CI/CD": ("ci/cd", "cicd"),
    "Linux": ("linux",),
    "AWS": ("aws", "amazon web services"),
    "GCP": ("gcp", "google cloud"),
    "Azure": ("azure",),
    "PostgreSQL": ("postgresql", "postgres"),
    "MySQL": ("mysql",),
    "MongoDB": ("mongodb",),
    "NoSQL": ("nosql",),
    "SQL Server": ("sql server",),
    "Redis": ("redis",),
    "Oracle": ("oracle",),
    "Agile/Scrum": ("agile", "scrum"),
    "Microservices": ("microservices", "microservice"),
    # Desain dan video
    "Photoshop": ("photoshop",),
    "Illustrator": ("illustrator",),
    "InDesign": ("indesign",),
    "Premiere Pro": ("premiere pro", "premiere", "adobe premiere"),
    "After Effects": ("after effects", "after effect"),
    "Adobe Creative Suite": ("adobe creative suite", "adobe creative cloud", "adobe cc"),
    "CorelDRAW": ("coreldraw", "corel draw", "corel"),
    "Figma": ("figma",),
    "Canva": ("canva",),
    "CapCut": ("capcut",),
    "Blender": ("blender",),
    "UI/UX": ("ui/ux", "ui", "ux", "user interface", "user experience"),
    "Motion Graphics": ("motion graphic", "motion graphics"),
    "Typography": ("typography", "tipografi"),
    "Video Editing": ("video editing", "editing video"),
    "Copywriting": ("copywriting", "copywriter"),
    # Marketing dan sales
    "SEO": ("seo",),
    "SEM": ("sem",),
    "Google Ads": ("google ads", "adwords"),
    "Meta Ads": ("meta ads", "facebook ads", "fb ads", "instagram ads"),
    "TikTok Ads": ("tiktok ads",),
    "Google Analytics": ("google analytics", "ga4"),
    "Meta Business Suite": ("meta business suite",),
    "Email Marketing": ("email marketing",),
    "Mailchimp": ("mailchimp",),
    "HubSpot": ("hubspot",),
    "CRM": ("crm", "salesforce"),
    "KOL Management": ("kol", "kols", "influencer marketing"),
    "Performance Marketing": ("performance marketing",),
    "Trade Marketing": ("trade marketing",),
    "ATL/BTL": ("atl", "btl"),
    "Market Research": ("market research", "riset pasar"),
    "Social Media Marketing": ("social media marketing", "social media management"),
    "Marketplace (Shopee/Tokopedia)": ("shopee", "tokopedia", "lazada", "tiktok shop"),
    # Perkantoran, keuangan, HR
    "Microsoft Word": ("microsoft word", "ms word"),
    "PowerPoint": ("powerpoint", "power point"),
    "Microsoft Office": ("microsoft office", "ms office"),
    "SAP": ("sap",),
    "ERP": ("erp",),
    "Financial Analysis": ("financial analysis", "analisis keuangan"),
    "Financial Modeling": ("financial modeling", "financial modelling"),
    "Budgeting": ("budgeting",),
    "Forecasting": ("forecasting",),
    "Perpajakan": ("perpajakan", "taxation", "pph", "ppn", "e-faktur"),
    "Financial Reporting": ("financial reporting", "financial statements", "laporan keuangan"),
    "PSAK/IFRS": ("psak", "ifrs"),
    "Recruitment": ("recruitment", "rekrutmen", "talent acquisition"),
    "Payroll": ("payroll", "penggajian"),
    "HRIS": ("hris",),
    "UU Ketenagakerjaan": ("uu ketenagakerjaan", "labor law", "labour law", "employment law"),
    "Employee Relations": ("employee relations",),
    "Project Management": ("project management", "pmp"),
    # Bahasa
    "English": ("english", "bahasa inggris"),
    "Mandarin": ("mandarin",),
    "Japanese": ("japanese", "bahasa jepang"),
    "Korean": ("korean", "bahasa korea"),
}

# Pengecualian yang terukur di data, bukan dugaan:
# - "West Java", "Central Java" adalah wilayah, bukan bahasa pemrograman.
# - "recruitment process/agency/FAQ/service/fee" adalah kalimat penutup iklan.
_NOT_AFTER: dict[str, tuple[str, ...]] = {"java": ("west", "east", "central", "north", "south")}
_NOT_BEFORE: dict[str, tuple[str, ...]] = {
    "recruitment": ("process", "agency", "faq", "service", "services", "fee", "fees"),
}

# Alias yang hanya aman di daftar skill CV: "Word" di CV pasti Microsoft Word, tetapi "word"
# di deskripsi lowongan adalah kata biasa (muncul di 29 lowongan).
CV_ONLY_ALIASES: dict[str, str] = {
    "word": "Microsoft Word",
    "ppt": "PowerPoint",
    "js": "JavaScript",
    "go": "Golang",
}

# Skill CV yang tidak dihitung (lihat docstring modul). Dibandingkan setelah huruf kecil.
SOFT_SKILLS = frozenset(
    {
        "communication",
        "communication skills",
        "komunikasi",
        "teamwork",
        "team work",
        "kerja sama tim",
        "collaboration",
        "leadership",
        "kepemimpinan",
        "problem solving",
        "problem-solving",
        "critical thinking",
        "berpikir kritis",
        "time management",
        "manajemen waktu",
        "adaptability",
        "creativity",
        "kreativitas",
        "public speaking",
        "negotiation",
        "negosiasi",
        "attention to detail",
        "interpersonal skills",
        "analytical thinking",
    }
)
# Keterangan tingkat di dalam kurung, misalnya "English (Intermediate)", bukan skill.
_LEVEL_WORDS = frozenset(
    {
        "beginner",
        "basic",
        "dasar",
        "intermediate",
        "menengah",
        "advanced",
        "mahir",
        "expert",
        "native",
        "fluent",
        "conversational",
        "professional",
        "proficient",
    }
)
_PARENS = re.compile(r"\(([^)]*)\)")
_SPLIT = re.compile(r"\s*[,;/]\s*")


def term_pattern(alias: str) -> re.Pattern[str]:
    """Regex satu alias: batas kata di kedua sisi, spasi/tanda hubung antarkata fleksibel.

    Batas kata memakai ``(?<!\\w)``/``(?!\\w)``, bukan ``\\b``, karena ``\\b`` gagal untuk
    alias yang berakhir simbol seperti "C++" atau "C#".
    """
    key = alias.strip().lower()
    tokens = re.split(r"[\s\-]+", key)
    body = r"[\s\-]*".join(re.escape(token) for token in tokens)
    prefix = "".join(rf"(?<!{word} )" for word in _NOT_AFTER.get(key, ()))
    suffix = ""
    if key in _NOT_BEFORE:
        suffix = r"(?![\s\-]+(?:" + "|".join(_NOT_BEFORE[key]) + r")(?!\w))"
    return re.compile(rf"{prefix}(?<!\w){body}(?!\w){suffix}", re.IGNORECASE)


def _vocab_pattern(aliases: Iterable[str]) -> re.Pattern[str]:
    return re.compile("|".join(term_pattern(alias).pattern for alias in aliases), re.IGNORECASE)


def _needles(aliases: Iterable[str]) -> tuple[str, ...]:
    """Potongan teks yang pasti ada jika alias cocok: token terpanjang setiap alias."""
    return tuple(max(re.split(r"[\s\-]+", alias.strip().lower()), key=len) for alias in aliases)


# (nama, needles, regex). Needles dicek dengan ``in`` sebelum regex: tanpa prefilter ini,
# indeks 473 lowongan butuh 3,9 dtk karena 123 regex dijalankan penuh di setiap deskripsi.
_VOCAB_INDEX: tuple[tuple[str, tuple[str, ...], re.Pattern[str]], ...] = tuple(
    (name, _needles(aliases), _vocab_pattern(aliases)) for name, aliases in SKILL_VOCAB.items()
)


def vocab_in_text(text: str | None) -> frozenset[str]:
    """Nama kosakata yang disebut di sebuah teks (deskripsi lowongan atau skill CV)."""
    if not text:
        return frozenset()
    lowered = text.lower()
    return frozenset(
        name
        for name, needles, pattern in _VOCAB_INDEX
        if any(needle in lowered for needle in needles) and pattern.search(text)
    )


def skill_variants(skill: str) -> list[str]:
    """Pecah satu skill CV menjadi varian yang bisa dicari di deskripsi.

    "Computer Vision (YOLO)" -> ["Computer Vision", "YOLO"]; "HTML/CSS" -> ["HTML", "CSS"];
    "English (Intermediate)" -> ["English"]. LLM menyalin skill persis dengan keterangan dalam
    kurung, sehingga string utuhnya hampir tidak pernah muncul di deskripsi lowongan.
    """
    inner = [part for group in _PARENS.findall(skill) for part in _SPLIT.split(group)]
    outer = _SPLIT.split(_PARENS.sub(" ", skill))
    variants: dict[str, str] = {}
    for part in [*outer, *inner]:
        text = " ".join(part.split())
        key = text.lower()
        if len(key) < 2 or key in _LEVEL_WORDS or key in SOFT_SKILLS:
            continue
        variants.setdefault(key, text)
    return list(variants.values())


@dataclass(frozen=True)
class CVSkills:
    """Skill CV yang sudah dipetakan; dibuat sekali per versi profil.

    Attributes:
        vocab: Nama kosakata yang dimiliki CV.
        extras: Varian skill CV yang tidak terpetakan ke kosakata, beserta regex-nya.
    """

    vocab: frozenset[str]
    extras: tuple[tuple[str, re.Pattern[str]], ...]


def build_cv_skills(skills: Sequence[str]) -> CVSkills:
    """Petakan skill profil CV ke kosakata; sisanya disimpan sebagai skill tambahan.

    Satu varian dianggap terpetakan jika menyebut minimal satu istilah kosakata, misalnya
    "Python (Pandas, NumPy)" -> Python, Pandas, NumPy. Varian yang tidak menyebut istilah
    kosakata ("YOLO", "SHAP") dicari apa adanya di deskripsi.
    """
    vocab: set[str] = set()
    extras: dict[str, tuple[str, re.Pattern[str]]] = {}
    for skill in skills:
        for variant in skill_variants(skill):
            alias = CV_ONLY_ALIASES.get(variant.lower())
            found = frozenset({alias}) if alias else vocab_in_text(variant)
            if found:
                vocab |= found
            else:
                extras.setdefault(variant.lower(), (variant, term_pattern(variant)))
    return CVSkills(vocab=frozenset(vocab), extras=tuple(extras.values()))


@dataclass(frozen=True)
class SkillMatch:
    """Hasil pencocokan skill untuk satu lowongan.

    Attributes:
        matched: Skill yang disebut lowongan dan dimiliki CV.
        missing: Skill kosakata yang disebut lowongan tetapi tidak ada di CV, paling umum
            di seluruh lowongan lebih dulu.
    """

    matched: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def required(self) -> int:
        """Jumlah skill yang disebut lowongan (kosakata + skill tambahan CV yang ditemukan)."""
        return len(self.matched) + len(self.missing)

    @property
    def portion(self) -> float | None:
        """Porsi skill lowongan yang dimiliki CV; ``None`` jika lowongan tidak menyebut skill."""
        return len(self.matched) / self.required if self.required else None


def match_skills(
    cv: CVSkills,
    job_vocab: Collection[str],
    description: str | None,
    *,
    frequency: Mapping[str, int] | None = None,
) -> SkillMatch:
    """Cocokkan skill CV dengan satu lowongan.

    Args:
        cv: Hasil ``build_cv_skills``.
        job_vocab: ``vocab_in_text(description)``; dihitung sekali per lowongan oleh
            pemanggil dan di-cache, karena tidak bergantung pada CV.
        description: Deskripsi lowongan, untuk mencari skill tambahan CV.
        frequency: Jumlah lowongan per nama kosakata, untuk mengurutkan skill. Tanpa ini,
            urutan alfabetis.

    Returns:
        ``SkillMatch``. Skill tambahan CV yang ditemukan dihitung sebagai disebut lowongan
        sekaligus dimiliki CV.
    """
    freq = frequency or {}

    def order(name: str) -> tuple[int, str]:
        return (-freq.get(name, 0), name.lower())

    matched = sorted((name for name in job_vocab if name in cv.vocab), key=order)
    missing = sorted((name for name in job_vocab if name not in cv.vocab), key=order)
    text = description or ""
    extras = [display for display, pattern in cv.extras if pattern.search(text)]
    return SkillMatch(matched=(*matched, *extras), missing=tuple(missing))
