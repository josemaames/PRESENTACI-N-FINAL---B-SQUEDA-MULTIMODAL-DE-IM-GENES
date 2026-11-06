"""Descarga Oxford-IIIT Pet (~3700 imágenes de gatos y perros) a ./images (opcional)."""
import shutil
from pathlib import Path

from torchvision.datasets import OxfordIIITPet

tmp = Path("_raw")
OxfordIIITPet(root=tmp, download=True)
out = Path("images")
out.mkdir(exist_ok=True)
for f in (tmp / "oxford-iiit-pet" / "images").glob("*.jpg"):
    shutil.copy(f, out / f.name)
shutil.rmtree(tmp)
print("Listo:", len(list(out.glob("*.jpg"))), "imágenes en ./images")
