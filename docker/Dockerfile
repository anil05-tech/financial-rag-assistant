FROM python:3.11-slim

WORKDIR /app

ENV MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "-m", "financial_rag"]
