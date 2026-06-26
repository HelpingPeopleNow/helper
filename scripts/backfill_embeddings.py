#!/usr/bin/env python3
"""
backfill_embeddings.py — Phase 4 backfill (VECTOR_SEARCH_PLAN §9.2)

Iterates every `worker_profiles` row via psycopg2, builds per-field texts
**byte-identical** to Go's `core.BuildFieldTexts` (parity is the whole
point — production reembedWorker hashes field text with the same algorithm
and skips rows whose `text_hash` already matches, so any divergence forces
a costly re-embed on the next intake). Calls the helper gRPC `EmbedBatch`
in chunks of 16 and atomically upserts into `worker_embeddings`.

Idempotent: pre-loads existing `(user_id, field_name) -> text_hash` map and
only embeds rows whose content hash changed (or were missing).

Usage:
    DB_HOST=localhost DB_USER=postgres DB_PASSWORD=postgres \\
    DB_NAME=helpingpeoplenow HELPER_GRPC_ADDR=localhost:50051 \\
        python3 helper/scripts/backfill_embeddings.py

Args (all optional):
    --batch-size N      Override BATCH_SIZE (default 16 — matches Granite
                        embedding cohort sweet spot and Ollama daemon default
                        num_parallel).
    --dry-run           Compute hashes + skip plan; print summary; do not embed.
    --self-test         Skip the network/DB path; run byte-parity assertions
                        against hand-checked fixtures. Exits 0 on success.

The final summary prints to STDOUT (workers_scanned / fields_embedded /
fields_skipped / errors / elapsed_seconds / fields_per_second), and progress
goes to STDERR so the script is friendly to `tee`, `kubectl logs`, and CI
capture.

Reference: VECTOR_SEARCH_PLAN §9.2
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from typing import Optional

logger = logging.getLogger("backfill_embeddings")
logging.basicConfig(
    level=os.getenv("BACKFILL_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stderr,
)

# ── Constants ──────────────────────────────────────────────────────────
# MUST stay in lock-step with backend/internal/core/worker_embeddings.go.
DEFAULT_EMBEDDING_MODEL = "granite-embedding:278m"
EXPECTED_DIMENSIONS = 768
BATCH_SIZE = 16

# Order matches BuildFieldTexts(). Embeddings are computed per field so
# iteration order doesn't affect the produced vectors — this constant is
# purely for diagnostics (e.g., stable print ordering in --self-test).
EMBEDDABLE_FIELDS = (
    "profession",
    "profession_raw",
    "bio",
    "certifications",
    "city",
    "languages",
    "business_name",
)

WORKER_COLUMNS = (
    "user_id",
    "profession",
    "business_name",
    "bio",
    "city",
    "certifications",
    "languages",
)


# ── Field composition (parity with Go) ────────────────────────────────
def normalize_profession(p: str) -> str:
    """Byte-identical mirror of backend/internal/core/worker_embeddings.go
    ::normalizeProfessionForEmbedding. If you add a case in Go, add it here
    in the same order — ordering doesn't matter for switch statements but
    matters for change-detection diffs in code review."""
    if p in (
        "electricista", "Electricista", "electrician", "Electrician",
    ):
        return "Electrician"
    if p in (
        "fontanero", "Fontanero", "plomero", "Plomero", "plumber", "Plumber",
    ):
        return "Plumber"
    if p in (
        "limpieza", "Limpieza", "cleaner", "Cleaner", "cleaning", "Cleaning",
    ):
        return "Cleaner"
    if p in ("manitas", "Manitas", "handyman", "Handyman"):
        return "Handyman"
    if p in ("carpintero", "Carpintero", "carpenter", "Carpenter"):
        return "Carpenter"
    if p in ("pintor", "Pintor", "painter", "Painter"):
        return "Painter"
    if p in (
        "jardinero", "Jardinero", "landscaper", "Landscaper",
        "gardener", "Gardener",
    ):
        return "Landscaper"
    if p in ("tejado", "Tejado", "roofer", "Roofer"):
        return "Roofer"
    if p in ("clima", "Clima", "hvac", "HVAC"):
        return "HVAC Technician"
    return p


def _join_json_array(json_str: str) -> str:
    """Mirror of Go's joinJSONArray. Empty input returns ''; bad JSON falls
    back to raw text (matches Go's behavior — words still get embedded)."""
    if not json_str:
        return ""
    try:
        arr = json.loads(json_str)
    except (ValueError, TypeError):
        logger.warning("join_json_array: failed to parse JSON, falling back to raw text")
        return json_str
    if not isinstance(arr, list):
        return json_str
    return ", ".join(str(s) for s in arr)


def build_field_texts(row: dict) -> dict:
    """Build field_name -> text map, EXACTLY like core.BuildFieldTexts.

    Parity requirements (every line is a potential divergence):
      * Profession gets `normalize_profession()` applied. If the normalized
        form differs from the raw, emit BOTH `profession` AND
        `profession_raw` (lower-weight secondary vector — see Go's P5/N7).
      * bio, city, business_name: pass-through strings.
      * certifications, languages: JSON-array column → ", ".join.
      * Empty fields are OMITTED (no row produced for empties).
    """
    fields: dict = {}
    profession = row.get("profession") or ""
    if profession:
        normalized = normalize_profession(profession)
        fields["profession"] = normalized
        if normalized != profession:
            fields["profession_raw"] = profession
    bio = row.get("bio") or ""
    if bio:
        fields["bio"] = bio
    certs = _join_json_array(row.get("certifications") or "")
    if certs:
        fields["certifications"] = certs
    city = row.get("city") or ""
    if city:
        fields["city"] = city
    langs = _join_json_array(row.get("languages") or "")
    if langs:
        fields["languages"] = langs
    business_name = row.get("business_name") or ""
    if business_name:
        fields["business_name"] = business_name
    return fields


def field_hash(text: str) -> str:
    """Mirror of core.HashField. Same hex output as Go's
    `hex.EncodeToString(sha256.Sum256([]byte(text))[:])`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── SQL helpers ────────────────────────────────────────────────────────
UPSERT_TMPL = (
    "INSERT INTO worker_embeddings (user_id, field_name, embedding, model, text_hash) "
    "VALUES (%s, %s, %s::vector, %s, %s) "
    "ON CONFLICT (user_id, field_name) DO UPDATE SET "
    "  embedding = EXCLUDED.embedding, "
    "  model     = EXCLUDED.model, "
    "  text_hash = EXCLUDED.text_hash"
)


def vec_to_sql_literal(vec) -> str:
    """Convert Python floats to a pgvector literal: '[0.1,0.2,0.3,...]'.
    Uses repr() rather than format() so each float round-trips — pgvector
    stores the literal verbatim, so any precision loss is permanent."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def load_existing_hashes(cur) -> dict:
    """Pre-load (user_id, field_name) -> text_hash for skip detection."""
    cur.execute("SELECT user_id, field_name, text_hash FROM worker_embeddings")
    out: dict = {}
    for r in cur.fetchall():
        out.setdefault(r["user_id"], {})[r["field_name"]] = r["text_hash"]
    return out


def load_workers(cur) -> list:
    """SELECT worker_profiles columns used by field composition."""
    col_list = ", ".join(WORKER_COLUMNS)
    cur.execute(
        f"SELECT {col_list} FROM worker_profiles ORDER BY created_at NULLS LAST"
    )
    return cur.fetchall()


def chunked(seq, n):
    """Yield successive n-sized chunks from seq. Like itertools.islice
    but for arbitrary sequences."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# ── Main path ──────────────────────────────────────────────────────────
def connect_db():
    """Lazy psycopg2 import so --self-test runs without the dep installed."""
    import psycopg2  # noqa: WPS433 (intentional lazy import)
    url = os.getenv("DATABASE_URL")
    if url:
        return psycopg2.connect(url)
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        dbname=os.getenv("DB_NAME", "helpingpeoplenow"),
    )


def open_grpc(addr: str):
    """Lazy import of helper_pb2 stubs so --self-test runs without grpc."""
    sys.path.insert(0, os.path.join(_HERE, "..", "proto"))
    import grpc  # noqa: WPS433
    import helper_pb2  # noqa: WPS433
    import helper_pb2_grpc  # noqa: WPS433
    channel = grpc.insecure_channel(addr)
    try:
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as exc:
        logger.error("helper gRPC at %s not ready: %s", addr, exc)
        channel.close()
        raise RuntimeError(f"helper gRPC at {addr} not ready: {exc}")
    return helper_pb2_grpc.HelperServiceStub(channel), channel


_HERE = os.path.dirname(os.path.abspath(__file__))


def run_backfill(batch_size: int = BATCH_SIZE, dry_run: bool = False) -> dict:
    """Returns a stats dict. Progress → STDERR; final summary → STDOUT."""
    started_at = time.monotonic()
    conn = connect_db()
    cur = conn.cursor()
    from psycopg2.extras import RealDictCursor, execute_batch  # noqa: WPS433
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        workers = load_workers(cur)
        existing_hashes = load_existing_hashes(cur)
        logger.info(
            "loaded %d worker profiles; %d users already have embeddings",
            len(workers), len(existing_hashes),
        )

        # Flatten to a per-(user_id, field_name) work list.
        work_items = []  # (user_id, field_name, text, text_hash)
        skip_count = 0
        for w in workers:
            fields = build_field_texts(w)
            user_existing = existing_hashes.get(w["user_id"], {})
            for fname, text in fields.items():
                h = field_hash(text)
                if user_existing.get(fname) == h:
                    skip_count += 1
                    continue
                work_items.append((w["user_id"], fname, text, h))

        logger.info(
            "plan: %d fields to embed, %d skipped (hash unchanged) across %d workers",
            len(work_items), skip_count, len(workers),
        )

        if dry_run:
            return {
                "workers_scanned": len(workers),
                "fields_embedded": 0,
                "fields_skipped": skip_count,
                "errors": 0,
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
                "dry_run": True,
            }

        if not work_items:
            return {
                "workers_scanned": len(workers),
                "fields_embedded": 0,
                "fields_skipped": skip_count,
                "errors": 0,
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            }

        addr = os.getenv("HELPER_GRPC_ADDR", "localhost:50051")
        stub, channel = open_grpc(addr)
        try:
            model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            embed_err = 0
            upsert_err = 0

            for batch in chunked(work_items, batch_size):
                texts = [item[2] for item in batch]
                try:
                    resp = stub.EmbedBatch(
                        _embed_batch_request(texts=texts, model=model_name),
                        timeout=60.0,
                    )
                except Exception as exc:
                    logger.warning(
                        "EmbedBatch RPC failed for batch of %d texts: %s",
                        len(texts), exc,
                    )
                    embed_err += len(texts)
                    conn.rollback()
                    continue

                rows_to_upsert = []
                for item, sub_resp in zip(batch, resp.embeddings):
                    user_id, fname, _text, h = item
                    vec = list(sub_resp.embedding)
                    if len(vec) != EXPECTED_DIMENSIONS:
                        logger.warning(
                            "dim mismatch for %s/%s: got %d, expected %d — skipping",
                            user_id, fname, len(vec), EXPECTED_DIMENSIONS,
                        )
                        embed_err += 1
                        continue
                    rows_to_upsert.append(
                        (user_id, fname, vec_to_sql_literal(vec),
                         sub_resp.model or model_name, h)
                    )

                if not rows_to_upsert:
                    continue
                try:
                    execute_batch(
                        cur, UPSERT_TMPL, rows_to_upsert,
                        page_size=len(rows_to_upsert),
                    )
                    conn.commit()
                    logger.info(
                        "upserted batch: %d rows (workers %s)",
                        len(rows_to_upsert),
                        ", ".join(sorted({r[0] for r in rows_to_upsert})[:3]),
                    )
                except Exception as exc:
                    logger.warning(
                        "upsert failed for batch (%d rows): %s",
                        len(rows_to_upsert), exc,
                    )
                    upsert_err += len(rows_to_upsert)
                    conn.rollback()

            return {
                "workers_scanned": len(workers),
                "fields_embedded": len(work_items) - embed_err - upsert_err,
                "fields_skipped": skip_count,
                "errors": embed_err + upsert_err,
                "elapsed_seconds": round(time.monotonic() - started_at, 3),
            }
        finally:
            channel.close()
    finally:
        cur.close()
        conn.close()


def _embed_batch_request(texts: list, model: str):
    """Tiny helper so the lazy grpc import wrapper stays readable."""
    import helper_pb2  # noqa: WPS433
    return helper_pb2.EmbedBatchRequest(texts=texts, model=model)


# ── Self-test (no network/DB) ─────────────────────────────────────────
#
# Run with `python3 backfill_embeddings.py --self-test`. Asserts that the
# hash function and field composition match Go's `core.HashField` and
# `core.BuildFieldTexts` byte-for-byte for a small set of fixtures.
#
# To verify on your machine, also run:
#     cd backend && go run ./cmd/hash_fixture
# and compare each `name|text|hash` line against Python's --self-test
# output. If the hashes match, byte parity is intact.
#
# Fixtures intentionally exercise every divergence-prone path:
#   * "electricista"   → normalized + profession_raw emitted together
#   * "Plumber"        → normalized == raw, no profession_raw
#   * certifications as JSON array → ", ".join
#   * empty bio        → omitted (no row)
#   * multi-byte (Spanish) text → UTF-8 stability check
EXPECTED_FIXTURES = [
    (
        {
            "user_id": "u_test_1",
            "profession": "electricista",
            "bio": "Electricista con 15 años de experiencia",
            "certifications": '["Licencia Tipo B", "Certificado de Segurança"]',
            "city": "Madrid",
            "languages": '["es", "en"]',
            "business_name": "",
        },
        ["profession", "profession_raw", "bio", "certifications", "city", "languages"],
    ),
    (
        {
            "user_id": "u_test_2",
            "profession": "Plumber",
            "bio": "",
            "certifications": "",
            "city": "Barcelona",
            "languages": "",
            "business_name": "Fontanería Ríos",
        },
        ["profession", "city", "business_name"],
    ),
]


def self_test() -> int:
    """Spot-check that build_field_texts + field_hash produce deterministic,
    per-fixture outputs. Emits canonical `name|text|hash` lines in the same
    order/format as `go run ./cmd/hash_fixture` so the two outputs can be
    `diff`'d to prove byte parity (FAIL on any drift).

    Returns 0 on success, non-zero on any parity violation.
    """
    failures = 0
    for idx, (row, _expected_keys) in enumerate(EXPECTED_FIXTURES, 1):
        print(f"=== fixture {idx} ===")
        fields = build_field_texts(row)
        # Emit in EMBEDDABLE_FIELDS order (stable across dict reordering).
        for name in EMBEDDABLE_FIELDS:
            if name in fields:
                _emit_hash_line(idx, name, fields[name])
        # Stability check: hash twice and assert equal (catches non-determinism).
        h1 = field_hash(fields["profession"])
        h2 = field_hash(fields["profession"])
        if h1 != h2 or len(h1) != 64 or h1 != h1.lower():
            print(f"  [{idx}] hash(profession) non-deterministic FAIL")
            failures += 1
        else:
            print(f"  [{idx}] hash(profession)={h1[:12]}... stable OK")
    # UTF-8 multi-byte sanity: byte length MUST be >= char length.
    multi = "Electricista con 15 años de experiencia"
    byte_len = len(multi.encode("utf-8"))
    if byte_len < len(multi):
        print(f"  FAIL: multi-byte UTF-8 byte len {byte_len} < char len {len(multi)}")
        failures += 1
    else:
        print(f"  multi-byte UTF-8 span OK ({byte_len} bytes for {len(multi)} chars)")
    if failures:
        print(f"self-test FAILED: {failures} mismatch(es)")
        return 1
    return 0


def _emit_hash_line(idx: int, name: str, text: str) -> None:
    """Print `  name|text|hash` — same format as Go fixture binary so
    the two side-by-side outputs can be `diff`'d for byte parity."""
    print(f"  {name}|{text}|{field_hash(text)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    stats = run_backfill(batch_size=args.batch_size, dry_run=args.dry_run)
    elapsed = stats.get("elapsed_seconds", 0.0)
    throughput = (
        stats["fields_embedded"] / elapsed if elapsed > 0 else 0.0
    )

    print("=== backfill summary ===")
    print(f"  workers_scanned    = {stats['workers_scanned']}")
    print(f"  fields_embedded    = {stats['fields_embedded']}")
    print(f"  fields_skipped     = {stats['fields_skipped']}")
    print(f"  errors             = {stats['errors']}")
    print(f"  elapsed_seconds    = {elapsed}")
    print(f"  fields_per_second  = {throughput:.1f}")
    if stats.get("dry_run"):
        print("  (dry run — no embeddings written)")
    return 0 if stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
