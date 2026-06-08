"""
Postgres-backed prompt repository adapter.

Reads the configured system prompt for the helper service from the
`system_prompts.helper_prompt` column — a singleton row where each
column is a different service's system prompt (extensible at the DB
level by adding new columns).
"""
import os
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from internal.ports.prompt_repository import PromptRepository, SystemPrompt


class PostgresPromptRepository(PromptRepository):
    def __init__(self, dsn: Optional[str] = None) -> None:
        self._dsn = dsn or self._build_dsn()

    @staticmethod
    def _build_dsn() -> str:
        host = os.getenv("DB_HOST", "postgres")
        port = os.getenv("DB_PORT", "5432")
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        dbname = os.getenv("DB_NAME", "helpingpeoplenow")
        return f"host={host} port={port} user={user} password={password} dbname={dbname}"

    def get_pizza_system_prompt(self) -> SystemPrompt:
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT helper_prompt
                    FROM system_prompts
                    WHERE id = 1
                    LIMIT 1
                    """
                )
                row = cur.fetchone()

        if row is None:
            return SystemPrompt(
                text=(
                    "You are a strict pizza-only assistant. "
                    "You ONLY answer questions that are about pizza — its ingredients, "
                    "history, recipes, cultural variations, preparation techniques, or anything "
                    "pizza-adjacent. If the question is NOT about pizza, politely refuse to answer "
                    "and explain that you can only discuss pizza."
                ),
                enforces_pizza_only=True,
            )

        return SystemPrompt(
            text=row["helper_prompt"],
            enforces_pizza_only=True,
        )
