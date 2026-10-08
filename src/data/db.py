"""Optional PostgreSQL persistence for trips and segment features."""
from pathlib import Path

from src.config import DATABASE_URL


def get_engine():
    if not DATABASE_URL:
        return None
    from sqlalchemy import create_engine

    return create_engine(DATABASE_URL)


def init_schema(engine=None):
    engine = engine or get_engine()
    if not engine:
        return
    from sqlalchemy import text

    schema_path = Path(__file__).resolve().parent.parent.parent / "sql" / "schema.sql"
    if not schema_path.exists():
        return
    with open(schema_path) as f:
        with engine.connect() as conn:
            for stmt in f.read().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
            conn.commit()


def save_features_to_db(df, engine=None):
    engine = engine or get_engine()
    if engine is None or df is None or df.empty:
        return
    cols = [c for c in df.columns if c != "segments"]
    df[cols].to_sql("segment_features", engine, if_exists="replace", index=False, method="multi")
