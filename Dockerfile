FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home app
USER app

# Di AWS Lambda, adapter meneruskan request ke uvicorn di port 8080 (bawaan adapter);
# di luar Lambda adapter tidak aktif dan docker-compose menimpa CMD per layanan.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 \
    /lambda-adapter /opt/extensions/lambda-adapter
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
