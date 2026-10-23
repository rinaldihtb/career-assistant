from datetime import datetime

from pydantic import BaseModel


class Job(BaseModel):
    id: int
    job_title: str
    company_name: str | None = None
    location: str | None = None
    work_type: str | None = None
    job_description: str | None = None

    salary_min: int | None = None
    salary_max: int | None = None
    salary_period: str | None = None
    salary_currency: str | None = None

    status: str = "OPEN"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class JobLocation(BaseModel):
    """Projection ringan untuk query lokasi lowongan."""

    id: int
    location: str
