FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    echo "Helper built with whisper support"

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 50051
EXPOSE 8084
EXPOSE 8085

CMD ["python", "main.py"]
