"""Carga en PostgreSQL los embeddings y rutas ya calculados (index/embeddings.npy + files.txt).

Uso:
    python migrate_to_db.py            # falla si la tabla ya tiene datos
    python migrate_to_db.py --reset    # borra la tabla `images` y la vuelve a cargar

La carga usa COPY binario y crea el índice HNSW al final (es mucho más rápido que indexar fila a fila).
"""
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from app import db

INDEX_DIR = Path(__file__).resolve().parent / "index"

CHUNK = 5000


def main(reset: bool):
    emb = np.load(INDEX_DIR / "embeddings.npy").astype(np.float32)
    names = (INDEX_DIR / "files.txt").read_text(encoding="utf-8").split("\n")
    assert len(emb) == len(names), f"{len(emb)} vectores vs {len(names)} rutas"
    print(f"Cargando {len(names):,} imágenes en {db.DATABASE_URL.split('@')[-1]}")

    with psycopg.connect(db.DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        if reset:
            conn.execute("DROP TABLE IF EXISTS images")
        conn.execute(db.SCHEMA)
        n = conn.execute("SELECT count(*) FROM images").fetchone()[0]
        if n:
            sys.exit(f"La tabla ya tiene {n:,} filas. Usa --reset para recargarla.")

        t = time.perf_counter()
        with conn.cursor() as cur:
            with cur.copy("COPY images (path, dataset, label, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
                copy.set_types(["text", "text", "text", "vector"])
                for i, (p, v) in enumerate(zip(names, emb), 1):
                    dataset, label = db.dataset_and_label(p)
                    copy.write_row((p, dataset, label, v))
                    if i % CHUNK == 0:
                        print(f"  {i:,}/{len(names):,}", end="\r")
        print(f"\nCopia lista en {time.perf_counter() - t:.1f} s. Creando índice HNSW…")

        t = time.perf_counter()
        conn.execute(db.INDEX_SQL)
        print(f"Índice HNSW creado en {time.perf_counter() - t:.1f} s")
        conn.execute("ANALYZE images")

        for ds, c in conn.execute("SELECT dataset, count(*) FROM images GROUP BY dataset ORDER BY 1"):
            print(f"  {ds}: {c:,}")
        print("Total:", conn.execute("SELECT count(*) FROM images").fetchone()[0])


if __name__ == "__main__":
    main("--reset" in sys.argv)
