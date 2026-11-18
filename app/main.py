"""Buscador multimodal de imágenes (texto->imagen e imagen->imagen) con CLIP.

- Imágenes: openai/clip-vit-base-patch32 (Hugging Face).
- Texto: sentence-transformers/clip-ViT-B-32-multilingual-v1, que proyecta texto en
  50+ idiomas (español incluido) al MISMO espacio que las imágenes de CLIP ViT-B/32,
  por lo que el índice de imágenes no cambia.
- Búsqueda: SEARCH_BACKEND=faiss (por defecto, índice en memoria) o
  SEARCH_BACKEND=postgres (PostgreSQL + pgvector, índice HNSW; ver docker-compose.yml).
"""
import io
import os
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

BASE = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE / "images"
INDEX_DIR = BASE / "index"
MODEL_NAME = "openai/clip-vit-base-patch32"
TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DEFAULT_K, MAX_K = 6, 24
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BACKEND = os.getenv("SEARCH_BACKEND", "faiss").lower()  # "faiss" | "postgres"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

_text_model = None  # carga perezosa: no hace falta para construir el índice


def _get_text_model():
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer

        _text_model = SentenceTransformer(TEXT_MODEL_NAME, device=device)
    return _text_model


def _norm(x: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.normalize(x, dim=-1).cpu().numpy().astype("float32")


@torch.no_grad()
def embed_images(imgs):
    inputs = processor(images=imgs, return_tensors="pt").to(device)
    out = model.vision_model(pixel_values=inputs["pixel_values"])
    return _norm(model.visual_projection(out.pooler_output))


@torch.no_grad()
def embed_text(text: str):
    v = _get_text_model().encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return v.astype("float32")


def build_index(batch_size: int = 32):
    files = sorted(p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No hay imágenes en {IMAGES_DIR}")
    vecs, names = [], []
    for i in range(0, len(files), batch_size):
        batch = files[i : i + batch_size]
        imgs, ok = [], []
        for f in batch:
            try:
                imgs.append(Image.open(f).convert("RGB"))
                ok.append(f)
            except Exception:
                pass
        vecs.append(embed_images(imgs))
        names += [str(f.relative_to(IMAGES_DIR)).replace("\\", "/") for f in ok]
        print(f"{len(names)}/{len(files)}", end="\r")
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", np.concatenate(vecs))
    (INDEX_DIR / "files.txt").write_text("\n".join(names), encoding="utf-8")
    print(f"\nIndexadas {len(names)} imágenes")


# --- Índice en memoria (solo backend "faiss": FAISS si está instalado, si no numpy) ---
emb = names = faiss_index = None
name_to_idx = {}
if BACKEND == "postgres":
    from app import db
else:
    if (INDEX_DIR / "embeddings.npy").exists():
        emb = np.load(INDEX_DIR / "embeddings.npy")
        names = (INDEX_DIR / "files.txt").read_text(encoding="utf-8").split("\n")
        name_to_idx = {n: i for i, n in enumerate(names)}
        try:
            import faiss

            faiss_index = faiss.IndexFlatIP(emb.shape[1])
            faiss_index.add(emb)
        except ImportError:
            pass


def _search_memory(q: np.ndarray, k: int, skip: int | None = None):
    kk = k + 1 if skip is not None else k
    if faiss_index is not None:
        scores, idx = faiss_index.search(q, kk)
        scores, idx = scores[0], idx[0]
    else:
        s = emb @ q[0]
        idx = np.argsort(-s)[:kk]
        scores = s[idx]
    out = [
        {"url": f"/images/{names[i]}", "name": names[i], "score": round(float(s), 4)}
        for i, s in zip(idx, scores)
        if i != skip
    ]
    return out[:k]


def search(q: np.ndarray, k: int = DEFAULT_K, skip_name: str | None = None):
    """Devuelve las k imágenes más cercanas al vector q, con el backend configurado."""
    k = max(1, min(int(k), MAX_K))
    if BACKEND == "postgres":
        rows = db.search(q, k, skip_path=skip_name)
        return [{"url": f"/images/{p}", "name": p, "score": round(float(s), 4)} for p, s in rows]
    return _search_memory(q, k, name_to_idx.get(skip_name) if skip_name else None)


def _ready() -> bool:
    return BACKEND == "postgres" or emb is not None


def _not_ready():
    msg = "Base de datos sin datos o inaccesible. Ejecuta: docker compose up -d y python migrate_to_db.py" if BACKEND == "postgres" \
        else "Índice no construido. Ejecuta: python -m app.main --build"
    return JSONResponse({"error": msg}, 503)


app = FastAPI(title="Image Search CLIP")


@app.on_event("startup")
def _warmup():
    """Carga el modelo de texto al arrancar para que la primera búsqueda sea rápida."""
    try:
        embed_text("warmup")
    except Exception as e:
        print("Aviso: no se pudo precargar el modelo de texto:", e)
    if BACKEND == "postgres":
        try:
            db.init_pool()
            print(f"PostgreSQL listo: {db.count():,} imágenes")
        except Exception as e:
            print("Aviso: no se pudo conectar a PostgreSQL:", e)


@app.get("/api/info")
def info():
    if BACKEND == "postgres":
        try:
            return {"images": db.count(), "index": "PostgreSQL + pgvector (HNSW)"}
        except Exception:
            return {"images": 0, "index": "PostgreSQL (sin conexión)"}
    return {"images": 0 if emb is None else len(emb), "index": "faiss" if faiss_index is not None else "numpy"}


@app.get("/api/search/text")
def search_text(q: str, k: int = DEFAULT_K):
    if not _ready():
        return _not_ready()
    t = time.perf_counter()
    try:
        res = search(embed_text(q), k)
    except Exception as e:
        return JSONResponse({"error": f"Error al buscar: {e}"}, 503)
    return {"results": res, "ms": round((time.perf_counter() - t) * 1000, 1)}


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...), k: int = Form(DEFAULT_K)):
    if not _ready():
        return _not_ready()
    t = time.perf_counter()
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    try:
        res = search(embed_images([img]), k)
    except Exception as e:
        return JSONResponse({"error": f"Error al buscar: {e}"}, 503)
    return {"results": res, "ms": round((time.perf_counter() - t) * 1000, 1)}


@app.get("/api/search/similar")
def search_similar(name: str, k: int = DEFAULT_K):
    if not _ready():
        return _not_ready()
    t = time.perf_counter()
    try:
        if BACKEND == "postgres":
            v = db.get_vector(name)
            if v is None:
                return JSONResponse({"error": "Imagen no encontrada en el índice."}, 404)
            res = search(v[None, :], k, skip_name=name)
        else:
            i = name_to_idx.get(name)
            if i is None:
                return JSONResponse({"error": "Imagen no encontrada en el índice."}, 404)
            res = search(emb[i : i + 1], k, skip_name=name)
    except Exception as e:
        return JSONResponse({"error": f"Error al buscar: {e}"}, 503)
    return {"results": res, "ms": round((time.perf_counter() - t) * 1000, 1)}


IMAGES_DIR.mkdir(exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/")
def root():
    return FileResponse(BASE / "static" / "index.html")


if __name__ == "__main__":
    import sys

    if "--build" in sys.argv:
        build_index()
