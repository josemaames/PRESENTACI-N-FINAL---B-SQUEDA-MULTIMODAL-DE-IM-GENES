"""Compara dos métodos de búsqueda sobre las mismas 12 390 imágenes: CLIP ViT-B/32 y SigLIP ViT-B/16.

Uso (necesita index/embeddings.npy y index/embeddings_siglip.npy; ver README):
    python compare_methods.py

Qué mide (todo con búsqueda exacta, así la única diferencia es el modelo que genera los vectores):
  A. Imagen -> imagen (Oxford-IIIT Pet): precisión@6 = fracción de los 6 resultados de la misma raza que la consulta.
  B. Texto -> imagen (Oxford-IIIT Pet): una consulta por raza (37), precisión@6 y @20 contra la raza correcta;
     más consultas "un gato/un perro" frente a "a cat/a dog" para ver el español (acierto de especie).
  C. Acuerdo entre métodos: cuántos de los 6 resultados comparten ambos con consultas de texto variadas
     (incluye COCO, que no tiene etiquetas) y con imágenes de consulta de ambos datasets.
  D. Costos: dimensión, tamaño del índice, tiempo de codificar una imagen y un texto, tiempo de búsqueda.
Escribe docs/comparacion_resultados.json y docs/comparacion_cualitativa.png.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from app import main as M

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
K = 6
rng = np.random.default_rng(0)

names = M.names
E = {"clip": np.load(M.INDEX_DIR / "embeddings.npy"), "siglip": np.load(M.INDEX_DIR / "embeddings_siglip.npy")}
assert len(names) == len(E["clip"]) == len(E["siglip"]), "los dos índices deben tener las mismas imágenes"
is_coco = np.array([n.startswith("coco/") for n in names])
labels = np.array([("" if c else n.rsplit("/", 1)[-1].rsplit(".", 1)[0].rsplit("_", 1)[0]) for n, c in zip(names, is_coco)])
pet_idx = np.where(~is_coco)[0]
breeds = sorted(set(labels[pet_idx]))
is_cat = np.array([bool(l) and l[0].isupper() for l in labels])  # en Oxford-IIIT Pet los gatos van con mayúscula


# Variantes de TEXTO: "clip" = el de la app (multilingüe, permite español), "clip_en" = el texto original de
# CLIP (solo inglés), "siglip". Las dos de CLIP buscan sobre los mismos vectores de imagen.
TV = ["clip", "clip_en", "siglip"]
IMG = {"clip_en": "clip"}


@torch.no_grad()
def embed_text(q, m):
    if m == "clip_en":
        inp = M.processor.tokenizer([q], padding=True, return_tensors="pt")
        return M._norm(M.model.text_projection(M.model.text_model(**inp).pooler_output))
    return M.embed_text(q, m)


def topk(method, q, k, skip=None):
    s = E[IMG.get(method, method)] @ q
    if skip is not None:
        s[skip] = -9
    return np.argsort(-s)[:k]


def overlap(a, b):
    return len(set(a) & set(b)) / len(a)


res = {"k": K, "images": len(names)}

# ---- A. imagen -> imagen (precisión por raza) ----
sample = rng.choice(pet_idx, 1500, replace=False)
res["img2img_precision@6"] = {
    m: float(np.mean([np.mean(labels[topk(m, E[m][i], K, i)] == labels[i]) for i in sample])) for m in E
}

# ---- B. texto -> imagen ----
def breed_prompt(b):
    return f"a photo of a {b.replace('_', ' ')}"


txt = {m: {b: embed_text(breed_prompt(b), m)[0] for b in breeds} for m in TV}
for kk in (6, 20):
    res[f"text2img_breed_precision@{kk}"] = {
        m: float(np.mean([np.mean(labels[topk(m, txt[m][b], kk)] == b) for b in breeds])) for m in TV
    }
species = {"cat": ("a cat", "un gato"), "dog": ("a dog", "un perro")}
res["text2img_species_precision@20"] = {}
for lang_i, lang in enumerate(("en", "es")):
    for m in TV:
        if m == "clip_en" and lang == "es":
            continue  # el texto original de CLIP no entiende español
        accs = []
        for sp, prompts in species.items():
            idx = topk(m, embed_text(prompts[lang_i], m)[0], 20)
            accs.append(np.mean((is_cat[idx] & ~is_coco[idx]) if sp == "cat" else (~is_cat[idx] & ~is_coco[idx])))
        res["text2img_species_precision@20"].setdefault(m, {})[lang] = float(np.mean(accs))

# ---- C. acuerdo entre métodos ----
QUERIES = [
    "a sleeping cat", "un gato durmiendo", "a dog running on grass", "un perro corriendo en el pasto",
    "a person riding a horse", "una persona montando a caballo", "a plate of food", "un plato de comida",
    "a man playing tennis", "un jugador de tenis", "a red double decker bus", "un autobús rojo en la ciudad",
    "a pizza on a table", "una pizza sobre una mesa", "a person surfing a wave", "alguien surfeando",
    "a kitchen with a refrigerator", "una cocina moderna", "an airplane in the sky", "un avión en el cielo",
    "a giraffe", "una jirafa", "a skateboarder", "un perro con un frisbee", "a vase with flowers",
    "un tren en una estación", "a bowl of fruit", "un elefante", "a laptop on a desk", "una moto en la calle",
]
qvec = {m: [embed_text(q, m)[0] for q in QUERIES] for m in TV}
agree_t = [overlap(topk("clip", qvec["clip"][j], K), topk("siglip", qvec["siglip"][j], K)) for j in range(len(QUERIES))]
EN_ONLY = [j for j, q in enumerate(QUERIES) if not any(c in q for c in "áéíóúñ") and q.split()[0] in ("a", "an")]
agree_t_en = [overlap(topk("clip_en", qvec["clip_en"][j], K), topk("siglip", qvec["siglip"][j], K)) for j in EN_ONLY]
coco_idx = np.where(is_coco)[0]
qs = np.concatenate([rng.choice(pet_idx, 150, replace=False), rng.choice(coco_idx, 150, replace=False)])
agree_i = [overlap(topk("clip", E["clip"][i], K, i), topk("siglip", E["siglip"][i], K, i)) for i in qs]
res["agreement@6"] = {"text": float(np.mean(agree_t)), "text_clip_en_only_english_queries": float(np.mean(agree_t_en)),
                      "image": float(np.mean(agree_i))}

# ---- D. costos ----
probe = [Image.open(M.IMAGES_DIR / n).convert("RGB") for n in names[:32:2]]
cost = {}
for m in E:
    M.embed_images(probe[:2], m)  # calentamiento
    t = time.perf_counter()
    M.embed_images(probe, m)
    t_img = (time.perf_counter() - t) / len(probe) * 1000
    M.embed_text("warm", m)
    t = time.perf_counter()
    for q in QUERIES[:10]:
        M.embed_text(q, m)
    t_txt = (time.perf_counter() - t) / 10 * 1000
    t = time.perf_counter()
    for j in range(100):
        topk(m, E[m][j], K)
    t_srch = (time.perf_counter() - t) / 100 * 1000
    cost[m] = {"dim": int(E[m].shape[1]), "index_mb": round(E[m].nbytes / 1e6, 1), "encode_image_ms": round(t_img, 1),
               "encode_text_ms": round(t_txt, 1), "search_ms": round(t_srch, 2)}
res["cost"] = cost

DOCS.mkdir(exist_ok=True)
(DOCS / "comparacion_resultados.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


# ---- Figura cualitativa: mismas consultas, resultados de cada método ----
def figure(queries, path, thumb=150):
    try:
        font = ImageFont.truetype("arial.ttf", 17)
        small = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = small = ImageFont.load_default()
    label_w, row_h, gap = 130, thumb + 10, 8
    W = label_w + K * (thumb + gap)
    H = len(queries) * (24 + 2 * row_h) + 10
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    y = 5
    for q in queries:
        d.text((8, y), f"“{q}”", fill=(30, 30, 60), font=font)
        y += 24
        for m, tag in (("clip", "CLIP (app)"), ("siglip", "SigLIP")):
            d.text((8, y + thumb // 2 - 8), tag, fill=(90, 80, 220) if m == "clip" else (20, 140, 110), font=small)
            for c, i in enumerate(topk(m, embed_text(q, m)[0], K)):
                im = Image.open(M.IMAGES_DIR / names[i]).convert("RGB")
                im.thumbnail((thumb, thumb))
                canvas.paste(im, (label_w + c * (thumb + gap), y))
            y += row_h
    canvas.save(path)


figure(["a sleeping cat", "un perro corriendo en el pasto", "a person riding a horse", "una pizza sobre una mesa"],
       DOCS / "comparacion_cualitativa.png")

# ---- Resumen ----
p = lambda x: "—" if x is None else f"{100 * x:.1f} %"
print()
print("Comparación en 12 390 imágenes, top 6, búsqueda exacta")
print("CLIP (app) = texto multilingüe de la app | CLIP orig. = texto original de CLIP (solo inglés)")
print()
print(f"{'Métrica':<46}{'CLIP (app)':>12}{'CLIP orig.':>12}{'SigLIP':>10}")
print("-" * 80)
r = res["img2img_precision@6"]
print(f"{'Imagen->imagen, precisión@6 (misma raza)':<46}{p(r['clip']):>12}{p(r['clip']):>12}{p(r['siglip']):>10}")
for kk in (6, 20):
    r = res[f"text2img_breed_precision@{kk}"]
    lbl = f"Texto->imagen, precisión@{kk} (raza correcta)"
    print(f"{lbl:<46}{p(r['clip']):>12}{p(r['clip_en']):>12}{p(r['siglip']):>10}")
for lang, nm in (("en", "inglés"), ("es", "español")):
    r = {m: res["text2img_species_precision@20"].get(m, {}).get(lang) for m in TV}
    lbl = f"Gato/perro en {nm}, acierto@20"
    print(f"{lbl:<46}{p(r['clip']):>12}{p(r['clip_en']):>12}{p(r['siglip']):>10}")
a = res["agreement@6"]
print()
print(f"Acuerdo con SigLIP (overlap@6): CLIP app, texto {p(a['text'])} | CLIP orig. (consultas en inglés) {p(a['text_clip_en_only_english_queries'])} | imagen {p(a['image'])}")
print()
print(f"{'Costo':<30}{'CLIP':>12}{'SigLIP':>12}")
for key, label in (("dim", "dimensiones"), ("index_mb", "índice (MB)"), ("encode_image_ms", "codificar imagen (ms)"),
                   ("encode_text_ms", "codificar texto (ms)"), ("search_ms", "buscar (ms)")):
    print(f"{label:<30}{cost['clip'][key]:>12}{cost['siglip'][key]:>12}")
