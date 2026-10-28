from src.models.agent import LLMUsage


def format_number(value: int) -> str:
    """Format angka Indonesia: ``2400`` -> ``"2.400"``."""
    return f"{value:,}".replace(",", ".")


def format_cost(cost_usd: float | None) -> str:
    """Biaya dalam USD dengan 6 desimal; ``"-"`` jika harga model tidak diketahui."""
    return "-" if cost_usd is None else f"${cost_usd:.6f}"


def format_usage(usage: LLMUsage) -> str:
    """Ringkasan pemakaian LLM: jumlah panggilan, model, token masuk dan keluar, serta biaya.

    Contoh: ``"1 panggilan LLM (gpt-4o-mini-2024-07-18) · 1.000 token masuk, 200 keluar ·
    $0.000270"``.
    """
    models = ", ".join(dict.fromkeys(call.model for call in usage.calls))
    calls = f"{usage.llm_calls} panggilan LLM" + (f" ({models})" if models else "")
    tokens_in = format_number(usage.input_tokens)
    tokens_out = format_number(usage.output_tokens)
    cost = format_cost(usage.cost_usd)
    return f"{calls} · {tokens_in} token masuk, {tokens_out} keluar · {cost}"


def format_salary(salary_min: int | None, salary_max: int | None) -> str:
    """Format gaji Rupiah per bulan; ``"tidak dicantumkan"`` jika kosong.

    Contoh: ``(8_000_000, 10_000_000)`` -> ``"Rp8.000.000 - Rp10.000.000 per bulan"``.
    Teks eksplisit "tidak dicantumkan" lebih aman daripada menghapus baris Gaji:
    LLM tidak tergoda menebak angka.
    """

    def rupiah(value: int) -> str:
        return f"Rp{value:,}".replace(",", ".")

    if salary_min is None and salary_max is None:
        return "tidak dicantumkan"
    if salary_min is not None and salary_max is not None:
        if salary_min != salary_max:
            return f"{rupiah(salary_min)} - {rupiah(salary_max)} per bulan"
        return f"{rupiah(salary_min)} per bulan"

    salary = salary_min if salary_min is not None else salary_max
    if salary is None:
        return "tidak dicantumkan"
    return f"{rupiah(salary)} per bulan"
