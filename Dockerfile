FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# R7: explicit COPY (not COPY . .) — only what the server needs.
COPY main.py ./
COPY internal ./internal
COPY proto ./proto

# R7: non-root user.
RUN useradd --create-home --uid 10001 helper && chown -R helper:helper /app
USER helper

# /app on path lets `from internal.adapters...` and `from proto import ...` resolve.
# /app/proto on path ALSO matters — the protoc-generated helper_pb2_grpc.py begins
# with `import helper_pb2` (not relative), and without /app/proto on PYTHONPATH
# the grpc server fails at import with ModuleNotFoundError. The /app/proto entry
# mirrors the sys.path hack in helper/scripts/backfill_embeddings.py for parity.
ENV PYTHONPATH=/app:/app/proto
ENV PYTHONUNBUFFERED=1

EXPOSE 50051
EXPOSE 8084

# R7: container-level liveness (uses the cached /health).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8084/health',timeout=4).status==200 else 1)"

CMD ["python", "main.py"]
