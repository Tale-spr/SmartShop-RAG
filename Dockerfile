FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir python-dotenv==1.1.1

COPY src ./src
RUN pip install --no-cache-dir -e .

COPY config ./config
COPY data ./data
COPY prompts ./prompts

EXPOSE 8000

CMD ["uvicorn", "smartshop_rag.api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
