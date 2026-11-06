"""Prepara ./deploy: copia app, static e index, y guarda las imágenes reducidas a 512 px."""
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deploy"
MAX = 512

for d in ("app", "static", "index"):
    shutil.copytree(ROOT / d, OUT / d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))

names = (ROOT / "index" / "files.txt").read_text(encoding="utf-8").split("\n")
for i, n in enumerate(names, 1):
    dst = OUT / "images" / n
    if dst.exists():
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(ROOT / "images" / n).convert("RGB")
    im.thumbnail((MAX, MAX))
    im.save(dst, "JPEG", quality=82, optimize=True)
    if i % 500 == 0:
        print(f"{i}/{len(names)}", end="\r")

size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
print(f"\nListo: {len(names)} imágenes. Tamaño de deploy/: {size:.0f} MB")
