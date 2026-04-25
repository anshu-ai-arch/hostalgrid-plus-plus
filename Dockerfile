FROM python:3.10-slim

LABEL name="hostalgrid-plus-plus"
LABEL description="EnergyMind - Human-Aware Energy Governance Environment"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN echo "/app" > /usr/local/lib/python3.10/site-packages/hostalgrid.pth

ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
