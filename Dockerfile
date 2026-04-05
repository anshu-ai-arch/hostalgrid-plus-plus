FROM python:3.8-slim

LABEL name="hostalgrid-plus-plus"
LABEL description="Human-Aware Energy Optimization - OpenEnv"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""

CMD ["python", "inference.py"]