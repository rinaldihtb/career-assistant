import os
from typing import Dict, Any
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langgraph_supervisor import create_supervisor

# 1. Mengimpor konfigurasi model, prompt global, dan database dari proyek
from src.app.config import CHAT_MODEL, SUPERVISOR_PROMPT
from config.llm import LLM_MODEL 

# 2. Mengimpor file agen fungsional milik rekan tim
from agents.sql_agent import run_sql_agent
# Catatan: Jika rekan tim sudah membuat agen vektor resume, silakan aktifkan baris di bawah ini:
# from agents.vector_agent import run_vector_agent 

# Jalur database SQLite lokal yang didefinisikan di environment
DB_PATH = os.getenv("DB_SQLITE_PATH", "./data/database.sqlite")


# ===========================================================================
# 1. WRAPPER AGENT (PEMBUNGKUS MODUL FUNGSIONAL TIM)
# ===========================================================================

def sql_agent_wrapper(state: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    """
    Menghubungkan Supervisor Agent (LangGraph) dengan fungsi run_sql_agent milik tim.
    Mengambil pertanyaan terakhir dari user dan meneruskan hasil kustom AgentResult.
    """
    messages = state.get("messages", [])
    user_question = ""
    
    # Mencari pesan bertipe user terakhir di dalam riwayat percakapan grafik
    for msg in reversed(messages):
        if msg.role == "user" or isinstance(msg, HumanMessage):
            user_question = msg.content
            break
            
    # Mengeksekusi fungsi utama dari file agents/sql_agent.py milik rekan tim
    result = run_sql_agent(question=user_question, llm=LLM_MODEL, db_path=DB_PATH)
    
    # Mengembalikan response ke LangGraph state tanpa memutus skema AgentResult dari tim
    return {
        "messages": [AIMessage(content=result.answer)],
        "team_agent_result": result  # Objek AgentResult disisipkan di sini untuk dibaca oleh submit_prompt
    }

def search_agent_wrapper(state: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    """
    Menghubungkan Supervisor Agent dengan modul pencarian vektor resume (Qdrant) milik tim.
    """
    # Catatan: Integrasikan dengan fungsi kustom buatan tim jika sudah siap
    # messages = state.get("messages", [])
    # user_question = ...
    # result = run_vector_agent(question=user_question, llm=LLM_MODEL, ...)
    # return {"messages": [AIMessage(content=result.answer)], "team_agent_result": result}
    return {"messages": [AIMessage(content="Fitur pencarian resume sedang diproses oleh agen terkait.")]}


# ===========================================================================
# 2. LOGIKA ROUTING & PENENTU TIPE QUERY (DETERMINATION LOGIC)
# ===========================================================================

def tentukan_tipe_query(query: str) -> str:
    """
    Fungsi klasifikasi awal menggunakan LLM ringan untuk mendeteksi apakah input 
    bersifat 'UMUM' (obrolan ringan/sapaan) atau 'SPESIFIK' (mencari data resume/lowongan).
    Menghindari pemanggilan agen RAG secara berlebihan untuk menghemat biaya token API.
    """
    routing_prompt = (
        "Tugasmu adalah mengklasifikasikan pesan pengguna ke dalam salah satu dari dua kategori berikut:\n"
        "1. 'UMUM' : Sapaan (halo, pagi), ucapan terima kasih, selamat tinggal, atau obrolan ringan di luar konteks pekerjaan.\n"
        "2. 'SPESIFIK' : Pertanyaan yang mencari lowongan kerja, statistik gaji, mencari resume, keahlian kandidat, atau kualifikasi pekerjaan.\n\n"
        "Aturan: Balas HANYA dengan satu kata antara 'UMUM' or 'SPESIFIK' tanpa tanda baca atau penjelasan tambahan."
    )
    
    try:
        response = LLM_MODEL.invoke([
            {"role": "system", "content": routing_prompt},
            {"role": "user", "content": query}
        ])
        return response.content.strip().upper()
    except Exception:
        # Fallback aman: Jika LLM klasifikasi error, teruskan ke jalur RAG/SPESIFIK
        return "SPESIFIK"


# ===========================================================================
# 3. KOMPILASI WORKFLOW SUPERVISOR AGENT
# ===========================================================================

def create_app_supervisor():
    """
    Fungsi untuk merakit dan mengompilasi Supervisor Agent menggunakan LangGraph.
    Supervisor bertindak sebagai manajer koordinasi antar sub-agent wrapper.
    """
    workflow = create_supervisor(
        model=CHAT_MODEL,
        prompt=SUPERVISOR_PROMPT,
        # Supervisor mendelegasikan tugas ke wrapper agen tim, bukan langsung ke tool database
        agents=[sql_agent_wrapper, search_agent_wrapper] 
    )

    supervisor = workflow.compile()
    return supervisor


# ===========================================================================
# 4. SUBMIT PROMPT UTAMA (TERINTEGRASI ROUTING & ANTI-HALUSINASI)
# ===========================================================================

def submit_prompt(app, prompt, config=None):
    """
    Fungsi eksekusi utama. Mengecek tipe query terlebih dahulu. 
    Jika 'UMUM', langsung dijawab oleh Supervisor dengan proteksi anti-halusinasi data.
    Jika 'SPESIFIK', diteruskan ke alur kerja LangGraph Multi-Agent terintegrasi tim.
    """
    # JALUR A: Jalankan deteksi klasifikasi tipe input pengguna
    tipe_query = tentukan_tipe_query(prompt)
    
    # JALUR B: Penanganan langsung jika query bersifat obrolan 'UMUM' (Bypass Jalur Agen)
    if tipe_query == "UMUM":
        # Prompt diperketat agar AI menolak mengarang/berhalusinasi tentang data internal lowongan
        general_anti_hallucination_prompt = (
            "Anda adalah asisten karier untuk aplikasi RAG Resume dan Lowongan Kerja.\n"
            "Tugas Anda HANYA menjawab sapaan, ucapan terima kasih, atau obrolan ringan (chitchat) secara ramah.\n"
            "PANTANGAN KERAS: Jangan pernah menyebutkan nama perusahaan, statistik gaji, jumlah lowongan, "
            "atau kualifikasi kandidat tertentu di sini karena Anda tidak memegang data tersebut saat ini.\n"
            "Jika pengguna mulai menanyakan data lowongan/kandidat, arahkan mereka dengan ramah untuk bertanya "
            "secara lebih spesifik agar sistem bisa mengaktifkan pencarian database."
        )
        
        response_llm = LLM_MODEL.invoke([
            {"role": "system", "content": general_anti_hallucination_prompt},
            {"role": "user", "content": prompt}
        ])
        
        # Hitung token penggunaan dari metadata respons model untuk General Chat
        total_input = response_llm.response_metadata.get("usage_metadata", {}).get("input_tokens", 0)
        total_output = response_llm.response_metadata.get("usage_metadata", {}).get("output_tokens", 0)
        price = (total_input * 0.15 + total_output * 0.6) / 1_000_000
        
        return {
            "answer": response_llm.content,
            "price": round(price, 5),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "tool_messages": ["Query dialihkan langsung ke General Chat (Proteksi Anti-Halusinasi aktif)."]
        }
        
    # JALUR C: Penanganan jika query bersifat 'SPESIFIK' (Masuk ke sistem LangGraph Multi-Agent)
    result = app.invoke({"messages": [
        {"role": "user", "content": prompt},
    ]}, config=config)

    # Mengambil respons final dari node terakhir di grafik percakapan
    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0
    tool_messages = []

    # MENYATUKAN PELACAKAN TOKEN DENGAN LOGIKA INTERNAL TIM (LLMUsage):
    # Mengekstrak metadata 'team_agent_result' yang dikembalikan oleh wrapper agen tim
    if "team_agent_result" in result:
        team_result = result["team_agent_result"]
        
        # Mengambil data hitungan token langsung dari objek kustom tim (.usage)
        total_input_tokens = getattr(team_result.usage, "input_tokens", 0)
        total_output_tokens = getattr(team_result.usage, "output_tokens", 0)
        
        # Mengekstrak riwayat query SQL milik tim untuk kebutuhan transparansi log UI
        if hasattr(team_result, "sql") and team_result.sql:
            tool_messages.append(f"SQL Query Executed:\n{team_result.sql}")
        if hasattr(team_result, "rows") and team_result.rows:
            tool_messages.append(f"Fetched Rows Count: {len(team_result.rows)}")

    # Jika sub-agent tim tidak mengembalikan log kustom, hitung token cadangan dari skema pesan grafik
    if total_input_tokens == 0 and total_output_tokens == 0:
        for message in result["messages"]:
            if hasattr(message, "response_metadata") and "usage_metadata" in message.response_metadata:
                total_input_tokens += message.response_metadata["usage_metadata"].get("input_tokens", 0)
                total_output_tokens += message.response_metadata["usage_metadata"].get("output_tokens", 0)

    # Kalkulasi perkiraan biaya pemanggilan API OpenAI berdasarkan harga dasar GPT-4o-mini
    price = (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000

    # Menyaring pesan bertipe ToolMessage bawaan jika ada sub-agent yang memicu pesan terstandar
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_messages.append(message.content)

    # Mengembalikan paket data lengkap untuk disalurkan ke frontend web Streamlit
    return {
        "answer": answer,
        "price": round(price, 5),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }
