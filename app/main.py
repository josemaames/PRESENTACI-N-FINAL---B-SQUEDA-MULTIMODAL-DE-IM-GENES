"""Buscador multimodal de imágenes (texto->imagen e imagen->imagen) con CLIP.

- Imágenes: openai/clip-vit-base-patch32 (Hugging Face).
- Texto: sentence-transformers/clip-ViT-B-32-multilingual-v1, que proyecta texto en
  50+ idiomas (español incluido) al MISMO espacio que las imágenes de CLIP ViT-B/32,
  por lo que el índice de imágenes no cambia.
- Método 2 (comparación): google/siglip-base-patch16-256-multilingual (SigLIP), que codifica imagen y
  texto en un espacio propio de 768 dimensiones (índice aparte: index/embeddings_siglip.npy).
  La interfaz permite cambiar entre ambos métodos con un botón (parámetro `method`: clip | siglip).
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
SIGLIP_NAME = "google/siglip-base-patch16-256-multilingual"
SIGLIP_TEXT_LEN = 64  # SigLIP se entrenó con texto rellenado a longitud fija
METHODS = {
    "clip": {"label": "CLIP ViT-B/32", "dim": 512},
    "siglip": {"label": "SigLIP ViT-B/16", "dim": 768},
}
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


_siglip = None  # (modelo, procesador): carga perezosa, solo si se usa el método 2


def _get_siglip():
    global _siglip
    if _siglip is None:
        from transformers import AutoModel, AutoProcessor

        _siglip = (AutoModel.from_pretrained(SIGLIP_NAME).to(device).eval(), AutoProcessor.from_pretrained(SIGLIP_NAME))
    return _siglip


@torch.no_grad()
def embed_images(imgs, method: str = "clip"):
    if method == "siglip":
        m, proc = _get_siglip()
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        return _norm(m.get_image_features(pixel_values=inputs["pixel_values"]).pooler_output)
    inputs = processor(images=imgs, return_tensors="pt").to(device)
    out = model.vision_model(pixel_values=inputs["pixel_values"])
    return _norm(model.visual_projection(out.pooler_output))


@torch.no_grad()
def embed_text(text: str, method: str = "clip"):
    if method == "siglip":
        m, proc = _get_siglip()
        inputs = proc(text=[text], padding="max_length", max_length=SIGLIP_TEXT_LEN, truncation=True, return_tensors="pt").to(device)
        return _norm(m.get_text_features(**inputs).pooler_output)
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


def build_siglip_index(batch_size: int = 16, part_size: int = 800):
    """Indexa con SigLIP las mismas imágenes (y en el mismo orden) que ya están en index/files.txt.

    Guarda el avance en index/siglip_parts/ cada `part_size` imágenes: si se interrumpe (Ctrl+C, apagado),
    al volver a ejecutar el mismo comando continúa donde se quedó.
    """
    import shutil

    files = (INDEX_DIR / "files.txt").read_text(encoding="utf-8").split("\n")
    parts = INDEX_DIR / "siglip_parts"
    parts.mkdir(exist_ok=True)
    for start in range(0, len(files), part_size):
        part = parts / f"part_{start:06d}.npy"
        if part.exists():
            continue
        chunk, vecs = files[start : start + part_size], []
        for i in range(0, len(chunk), batch_size):
            imgs = [Image.open(IMAGES_DIR / f).convert("RGB") for f in chunk[i : i + batch_size]]
            vecs.append(embed_images(imgs, "siglip"))
            print(f"{start + min(i + batch_size, len(chunk))}/{len(files)}", end="\r", flush=True)
        np.save(part, np.concatenate(vecs))
    allv = np.concatenate([np.load(f) for f in sorted(parts.glob("part_*.npy"))])
    assert len(allv) == len(files), f"{len(allv)} vectores para {len(files)} imágenes"
    np.save(INDEX_DIR / "embeddings_siglip.npy", allv)
    shutil.rmtree(parts)
    print(f"\nSigLIP: {len(files)} imágenes indexadas")


# --- Índice en memoria (solo backend "faiss": FAISS si está instalado, si no numpy) ---
emb = names = faiss_index = None
name_to_idx = {}
siglip_emb = siglip_faiss = None  # el método 2 siempre se busca en memoria (aun con backend postgres)


def _faiss_for(arr):
    try:
        import faiss

        ix = faiss.IndexFlatIP(arr.shape[1])
        ix.add(arr)
        return ix
    except ImportError:
        return None


if (INDEX_DIR / "files.txt").exists():
    names = (INDEX_DIR / "files.txt").read_text(encoding="utf-8").split("\n")
    name_to_idx = {n: i for i, n in enumerate(names)}
if (INDEX_DIR / "embeddings_siglip.npy").exists():
    siglip_emb = np.load(INDEX_DIR / "embeddings_siglip.npy")
    siglip_faiss = _faiss_for(siglip_emb)
if BACKEND == "postgres":
    from app import db
elif (INDEX_DIR / "embeddings.npy").exists():
    emb = np.load(INDEX_DIR / "embeddings.npy")
    faiss_index = _faiss_for(emb)


def _search_memory(q: np.ndarray, k: int, skip: int | None = None, method: str = "clip"):
    arr, fx = (siglip_emb, siglip_faiss) if method == "siglip" else (emb, faiss_index)
    kk = k + 1 if skip is not None else k
    if fx is not None:
        scores, idx = fx.search(q, kk)
        scores, idx = scores[0], idx[0]
    else:
        s = arr @ q[0]
        idx = np.argsort(-s)[:kk]
        scores = s[idx]
    out = [
        {"url": f"/images/{names[i]}", "name": names[i], "score": round(float(s), 4)}
        for i, s in zip(idx, scores)
        if i != skip
    ]
    return out[:k]


def search(q: np.ndarray, k: int = DEFAULT_K, skip_name: str | None = None, method: str = "clip"):
    """Devuelve las k imágenes más cercanas al vector q, con el método y backend configurados."""
    k = max(1, min(int(k), MAX_K))
    if method == "clip" and BACKEND == "postgres":
        rows = db.search(q, k, skip_path=skip_name)
        return [{"url": f"/images/{p}", "name": p, "score": round(float(s), 4)} for p, s in rows]
    return _search_memory(q, k, name_to_idx.get(skip_name) if skip_name else None, method)


def _ready(method: str = "clip") -> bool:
    if method == "siglip":
        return siglip_emb is not None
    return BACKEND == "postgres" or emb is not None


def _not_ready(method: str = "clip"):
    if method == "siglip":
        msg = "Índice SigLIP no construido. Ejecuta: python -m app.main --build-siglip"
    elif BACKEND == "postgres":
        msg = "Base de datos sin datos o inaccesible. Ejecuta: docker compose up -d y python migrate_to_db.py"
    else:
        msg = "Índice no construido. Ejecuta: python -m app.main --build"
    return JSONResponse({"error": msg}, 503)


def _method(m: str) -> str:
    return m if m in METHODS else "clip"


app = FastAPI(title="Image Search CLIP")


@app.on_event("startup")
def _warmup():
    """Carga los modelos al arrancar para que la primera búsqueda de cada método sea rápida."""
    try:
        embed_text("warmup")
    except Exception as e:
        print("Aviso: no se pudo precargar el modelo de texto:", e)
    if siglip_emb is not None:
        try:
            embed_text("warmup", "siglip")
        except Exception as e:
            print("Aviso: no se pudo precargar SigLIP:", e)
    if BACKEND == "postgres":
        try:
            db.init_pool()
            print(f"PostgreSQL listo: {db.count():,} imágenes")
        except Exception as e:
            print("Aviso: no se pudo conectar a PostgreSQL:", e)


@app.get("/api/info")
def info():
    methods = [
        {"id": k, "label": v["label"], "available": k == "clip" or siglip_emb is not None} for k, v in METHODS.items()
    ]
    if BACKEND == "postgres":
        try:
            return {"images": db.count(), "index": "PostgreSQL + pgvector (HNSW)", "methods": methods}
        except Exception:
            return {"images": 0, "index": "PostgreSQL (sin conexión)", "methods": methods}
    return {
        "images": 0 if emb is None else len(emb),
        "index": "faiss" if faiss_index is not None else "numpy",
        "methods": methods,
    }


@app.get("/api/search/text")
def search_text(q: str, k: int = DEFAULT_K, method: str = "clip"):
    method = _method(method)
    if not _ready(method):
        return _not_ready(method)
    t = time.perf_counter()
    try:
        res = search(embed_text(q, method), k, method=method)
    except Exception as e:
        return JSONResponse({"error": f"Error al buscar: {e}"}, 503)
    return {"results": res, "ms": round((time.perf_counter() - t) * 1000, 1), "method": method}


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...), k: int = Form(DEFAULT_K), method: str = Form("clip")):
    method = _method(method)
    if not _ready(method):
        return _not_ready(method)
    t = time.perf_counter()
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    try:
        res = search(embed_images([img], method), k, method=method)
    except Exception as e:
        return JSONResponse({"error": f"Error al buscar: {e}"}, 503)
    return {"results": res, "ms": round((time.perf_counter() - t) * 1000, 1), "method": method}


@app.get("/api/search/similar")
def search_similar(name: str, k: int = DEFAULT_K, method: str = "clip"):
    method = _method(method)
    if not _ready(method):
        return _not_ready(method)
    t = time.perf_counter()
    try:
        if method == "clip" and BACKEND == "postgres":
            v = db.get_vector(name)
            if v is None:
                return JSONResponse({"error": "Imagen no encontrada en el índice."}, 404)
            res = search(v[None, :], k, skip_name=name)
        else:
            i = name_to_idx.get(name)
            if i is None:
                return JSONResponse({"error": "Imagen no encontrada en el índice."}, 404)
            arr = siglip_emb if method == "siglip" else emb
            res = search(arr[i : i + 1], k, skip_name=name, method=method)
    except Exception as e:
        return JSONResponse({"error": f"Error al buscar: {e}"}, 503)
    return {"results": res, "ms": round((time.perf_counter() - t) * 1000, 1), "method": method}


IMAGES_DIR.mkdir(exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/")
def root():
    return FileResponse(BASE / "static" / "index.html")


if __name__ == "__main__":
    import sys

    if "--build" in sys.argv:
        build_index()
    if "--build-siglip" in sys.argv:
        build_siglip_index()
