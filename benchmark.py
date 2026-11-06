"""Benchmark de escalabilidad: búsqueda exacta (numpy / FAISS Flat) vs aproximada (IVF / HNSW).

Uso:
    python benchmark.py          # 7 390, 100 mil y 500 mil vectores
    python benchmark.py --big    # agrega 1 millón (tarda varios minutos y usa ~2 GB de RAM)

Los índices más grandes que el dataset real se completan con vectores sintéticos creados
perturbando embeddings reales (así conservan la estructura de clusters de CLIP).
El costo de buscar depende del número de vectores, no de si son reales.
"""
import sys
import time

import faiss
import numpy as np

from app.main import embed_text, emb

K = 6
d = emb.shape[1]
rng = np.random.default_rng(0)

# Consultas: 5 de texto reales + 100 embeddings de imagen tomados del índice real
texts = ["a sleeping cat", "a dog running on grass", "a black cat", "a puppy", "two dogs"]
t = time.perf_counter()
text_q = np.concatenate([embed_text(q) for q in texts])
t_embed = (time.perf_counter() - t) / len(texts) * 1000
queries = np.concatenate([text_q, emb[rng.choice(len(emb), 100, replace=False)]]).astype("float32")
print(f"Embedding de una consulta de texto (CLIP): {t_embed:.1f} ms")
print(f"Consultas de prueba: {len(queries)} | dimensión: {d} | top-{K}\n")


def make_vectors(n):
    if n <= len(emb):
        return emb[:n].copy()
    extra = emb[rng.integers(0, len(emb), n - len(emb))] + 0.05 * rng.standard_normal((n - len(emb), d)).astype("float32")
    v = np.concatenate([emb, extra]).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def timeit(fn):
    t = time.perf_counter()
    for q in queries:
        fn(q)
    return (time.perf_counter() - t) / len(queries) * 1000


def recall(index, truth):
    _, ids = index.search(queries, K)
    return np.mean([len(set(a) & set(b)) / K for a, b in zip(ids, truth)])


sizes = [7_390, 100_000, 500_000] + ([1_000_000] if "--big" in sys.argv else [])
rows = []
for n in sizes:
    print(f"== N = {n:,} ==", flush=True)
    base = make_vectors(n)

    # Exacto: numpy
    t_np = timeit(lambda q: np.argpartition(-(base @ q), K)[:K])

    # Exacto: FAISS Flat (verdad de referencia)
    flat = faiss.IndexFlatIP(d)
    flat.add(base)
    t_flat = timeit(lambda q: flat.search(q[None, :], K))
    _, truth = flat.search(queries, K)

    # Aproximado: IVF
    nlist = int(4 * np.sqrt(n))
    quant = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    ivf.train(base[rng.choice(n, min(n, 50_000), replace=False)])
    ivf.add(base)
    b_ivf = time.perf_counter() - t0
    ivf.nprobe = 16
    t_ivf = timeit(lambda q: ivf.search(q[None, :], K))
    r_ivf = recall(ivf, truth)

    # Aproximado: HNSW
    hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 100
    t0 = time.perf_counter()
    hnsw.add(base)
    b_hnsw = time.perf_counter() - t0
    hnsw.hnsw.efSearch = 64
    t_hnsw = timeit(lambda q: hnsw.search(q[None, :], K))
    r_hnsw = recall(hnsw, truth)

    rows.append((n, t_np, t_flat, t_ivf, r_ivf, b_ivf, t_hnsw, r_hnsw, b_hnsw))
    del base, flat, ivf, hnsw

print("\nTiempo de búsqueda por consulta (ms) y recall@6 respecto a la búsqueda exacta\n")
print(f"{'N':>10} | {'numpy':>7} | {'Flat':>7} | {'IVF':>7} {'recall':>6} {'build s':>8} | {'HNSW':>7} {'recall':>6} {'build s':>8}")
print("-" * 86)
for n, a, b, c, rc, bc, e, re_, be in rows:
    print(f"{n:>10,} | {a:>7.2f} | {b:>7.2f} | {c:>7.2f} {rc:>6.2f} {bc:>8.1f} | {e:>7.2f} {re_:>6.2f} {be:>8.1f}")
