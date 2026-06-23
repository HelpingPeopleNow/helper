FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    echo "Helper built with whisper support"

COPY . .

# /app on path lets `from internal.adapters...` and `from proto import ...` resolve.
# /app/proto on path ALSO matters — the protoc-generated helper_pb2_grpc.py begins
# with `import helper_pb2` (not relative), and without /app/proto on PYTHONPATH
# the grpc server fails at import with ModuleNotFoundError. The /app/proto entry
# mirrors the sys.path hack in helper/scripts/backfill_embeddings.py for parity.
ENV PYTHONPATH=/app:/app/proto
ENV PYTHONUNBUFFERED=1

EXPOSE 50051
EXPOSE 8084
EXPOSE 8085

CMD ["python", "main.py"]
