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
