"""Acceso a PostgreSQL + pgvector: esquema, carga masiva y búsqueda por similitud coseno.

Tabla `images`:  id (BIGSERIAL) · path (única) · dataset · label · embedding vector(512)
Índice HNSW sobre `embedding` con distancia coseno (`<=>`), que es lo que usa la búsqueda.
"""
import os

import numpy as np
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://imagesearch:imagesearch@localhost:5433/imagesearch")
DIM = 512
EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "64"))

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS images (
    id        BIGSERIAL PRIMARY KEY,
    path      TEXT UNIQUE NOT NULL,
    dataset   TEXT NOT NULL,
    label     TEXT,
    embedding vector({DIM}) NOT NULL
);
"""
INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS images_embedding_hnsw ON images "
    "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
)

_pool: ConnectionPool | None = None


def _configure(conn):
    register_vector(conn)
    conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
    conn.commit()


def init_pool(min_size: int = 1, max_size: int = 8) -> ConnectionPool:
    """Pool de conexiones para el servidor web."""
    global _pool
    if _pool is None:
        # La extensión debe existir antes de que `register_vector` busque el tipo.
        import psycopg

        with psycopg.connect(DATABASE_URL, autocommit=True) as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        _pool = ConnectionPool(DATABASE_URL, min_size=min_size, max_size=max_size, configure=_configure, open=True)
    return _pool


def dataset_and_label(path: str) -> tuple[str, str | None]:
    """Deduce dataset y etiqueta a partir de la ruta relativa.

    coco/000000001.jpg   -> ("coco", None)
    Abyssinian_143.jpg   -> ("pets", "Abyssinian")   (la raza va antes del número)
    """
    if path.startswith("coco/"):
        return "coco", None
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return "pets", stem.rsplit("_", 1)[0] if "_" in stem else None


def count() -> int:
    with init_pool().connection() as c:
        return c.execute("SELECT count(*) FROM images").fetchone()[0]


def search(vec: np.ndarray, k: int, skip_path: str | None = None, dataset: str | None = None):
    """Devuelve [(path, score)] con score = similitud coseno (1 - distancia)."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    where, filt = [], []
    if skip_path is not None:
        where.append("path <> %s")
        filt.append(skip_path)
    if dataset is not None:
        where.append("dataset = %s")
        filt.append(dataset)
    sql = (
        "SELECT path, 1 - (embedding <=> %s) AS score FROM images "
        + ("WHERE " + " AND ".join(where) + " " if where else "")
        + "ORDER BY embedding <=> %s LIMIT %s"
    )
    params = [v, *filt, v, k]  # orden de los %s: score, filtros, ORDER BY, LIMIT
    with init_pool().connection() as c:
        return c.execute(sql, params).fetchall()


def get_vector(path: str) -> np.ndarray | None:
    with init_pool().connection() as c:
        row = c.execute("SELECT embedding FROM images WHERE path = %s", (path,)).fetchone()
    if row is None:
        return None
    v = row[0]
    v = v.to_numpy() if hasattr(v, "to_numpy") else v  # pgvector devuelve un objeto Vector
    return np.asarray(v, dtype=np.float32)
