"""Benchmark de PostgreSQL + pgvector (índice HNSW) a gran escala.

Uso:
    python benchmark_pg.py                     # 100 mil y 500 mil vectores
    python benchmark_pg.py --big               # agrega 1 millón (necesita ~4 GB para Docker y varios minutos)
    python benchmark_pg.py --sizes=200000      # tamaños a medida, separados por coma
    python benchmark_pg.py --mem=3GB           # maintenance_work_mem para construir el índice (por defecto 3GB)

Usa una tabla temporal `bench_vectors` (no toca la tabla `images`) y la borra al terminar cada tamaño.
Los vectores son los embeddings reales más copias sintéticas (reales con ruido de 0,05, normalizadas),
generadas por bloques para no ocupar memoria. La verdad de referencia (búsqueda exacta) se calcula
dentro de Postgres, sin índice. Recall@6 = fracción de los 6 resultados que coincide con la exacta.
"""
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from app import db

K, D = 6, 512
EF_SEARCH = (40, 64, 128)
ROOT = Path(__file__).resolve().parent


def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


emb = np.load(ROOT / "index" / "embeddings.npy").astype(np.float32)
rng = np.random.default_rng(0)
queries = emb[rng.choice(len(emb), 105, replace=False)]
MEM = arg("mem", "3GB")


def blocks(n, chunk=50_000):
    """Genera n vectores por bloques: los reales primero y luego sintéticos."""
    done = 0
    while done < n:
        m = min(chunk, n - done)
        if done < len(emb):
            v = emb[done : done + m]
            m = len(v)
        else:
            v = emb[rng.integers(0, len(emb), m)] + 0.05 * rng.standard_normal((m, D), dtype=np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
        yield v
        done += m


def timed_queries(conn):
    ids, ms = [], []
    for q in queries:
        t = time.perf_counter()
        rows = conn.execute("SELECT id FROM bench_vectors ORDER BY embedding <=> %s LIMIT %s", (q, K)).fetchall()
        ms.append((time.perf_counter() - t) * 1000)
        ids.append({r[0] for r in rows})
    return ids, np.array(ms)


def run(n):
    print(f"\n== N = {n:,} ==", flush=True)
    with psycopg.connect(db.DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("DROP TABLE IF EXISTS bench_vectors")
        conn.execute(f"CREATE UNLOGGED TABLE bench_vectors (id bigint, embedding vector({D}))")

        t = time.perf_counter()
        i = 0
        with conn.cursor() as cur, cur.copy("COPY bench_vectors (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["int8", "vector"])
            for block in blocks(n):
                for v in block:
                    cp.write_row((i, v))
                    i += 1
                print(f"  cargados {i:,}", end="\r", flush=True)
        t_copy = time.perf_counter() - t
        conn.execute("ANALYZE bench_vectors")
        print(f"  carga: {t_copy:.1f} s                    ")

        # Verdad de referencia: búsqueda exacta (recorrido secuencial, sin índice)
        conn.execute("SET enable_indexscan = off")
        truth, ms_exact = timed_queries(conn)
        conn.execute("SET enable_indexscan = on")
        print(f"  exacta (sin índice): {ms_exact.mean():.1f} ms", flush=True)

        # Índice HNSW
        conn.execute(f"SET maintenance_work_mem = '{MEM}'")
        conn.execute("SET max_parallel_maintenance_workers = 4")
        t = time.perf_counter()
        conn.execute("CREATE INDEX bench_hnsw ON bench_vectors USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)")
        t_build = time.perf_counter() - t
        size_tbl, size_idx = conn.execute("SELECT pg_table_size('bench_vectors'), pg_relation_size('bench_hnsw')").fetchone()
        print(f"  índice HNSW: {t_build:.1f} s | tabla {size_tbl/1e6:.0f} MB | índice {size_idx/1e6:.0f} MB", flush=True)

        out = []
        for ef in EF_SEARCH:
            conn.execute(f"SET hnsw.ef_search = {ef}")
            timed_queries(conn)  # calentamiento
            got, ms = timed_queries(conn)
            rec = float(np.mean([len(a & b) / K for a, b in zip(got, truth)]))
            out.append((ef, ms.mean(), np.percentile(ms, 95), rec))
            print(f"  ef_search={ef}: {ms.mean():.2f} ms (p95 {np.percentile(ms, 95):.2f}) recall {rec:.3f}", flush=True)
        conn.execute("DROP TABLE bench_vectors")
    return dict(n=n, copy=t_copy, exact=ms_exact.mean(), build=t_build, tbl=size_tbl, idx=size_idx, ef=out)


if __name__ == "__main__":
    sizes = [int(x) for x in arg("sizes", "").split(",") if x] or [100_000, 500_000] + ([1_000_000] if "--big" in sys.argv else [])
    res = [run(n) for n in sizes]

    print("\nTiempo por consulta (ms) y recall@6 de pgvector HNSW frente a la búsqueda exacta, en CPU\n")
    print(f"{'N':>10} | {'exacta':>8} | {'ef=40':>7} {'rec':>5} | {'ef=64':>7} {'rec':>5} | {'ef=128':>7} {'rec':>5} | {'build s':>8} | {'índice MB':>9}")
    print("-" * 100)
    for r in res:
        e = {ef: (m, rec) for ef, m, _, rec in r["ef"]}
        print(f"{r['n']:>10,} | {r['exact']:>8.1f} | {e[40][0]:>7.2f} {e[40][1]:>5.2f} | {e[64][0]:>7.2f} {e[64][1]:>5.2f} | "
              f"{e[128][0]:>7.2f} {e[128][1]:>5.2f} | {r['build']:>8.1f} | {r['idx']/1e6:>9.0f}")
