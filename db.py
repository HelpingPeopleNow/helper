import os

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER', 'postgres')}"
    f":{os.getenv('DB_PASSWORD', 'postgres')}"
    f"@{os.getenv('DB_HOST', 'postgres')}"
    f":{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME', 'helpingpeoplenow')}"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class PromptHelper(Base):
    __tablename__ = "prompt_helpers"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(255), default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


def get_system_prompt():
    """Fetch the system prompt from DB. If no 'system-prompt' entry exists, create one."""
    session = SessionLocal()
    try:
        record = session.query(PromptHelper).filter(PromptHelper.title == "system-prompt").first()
        if record:
            return record.content

        # Not found — save the default pizza prompt
        default = (
            "You are a strict pizza-only assistant. "
            "You ONLY answer questions that are about pizza — its ingredients, "
            "history, recipes, cultural variations, preparation techniques, or anything "
            "pizza-adjacent. If the question is NOT about pizza, politely refuse to answer "
            "and explain that you can only discuss pizza."
        )
        record = PromptHelper(
            title="system-prompt",
            content=default,
            category="system",
        )
        session.add(record)
        session.commit()
        return default
    finally:
        session.close()
