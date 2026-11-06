"""Descarga COCO val2017 (5000 imágenes variadas, ~780 MB) a ./images/coco
y las suma a las mascotas que ya están en ./images."""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "http://images.cocodataset.org/zips/val2017.zip"
zip_path = Path("val2017.zip")
out = Path("images") / "coco"
out.mkdir(parents=True, exist_ok=True)


def progress(blocks, bsize, total):
    done = blocks * bsize
    if total > 0:
        sys.stdout.write(f"\rDescargando: {min(done, total) / 1e6:.0f}/{total / 1e6:.0f} MB")
        sys.stdout.flush()


if not zip_path.exists():
    urllib.request.urlretrieve(URL, zip_path, progress)
    print()

n = 0
with zipfile.ZipFile(zip_path) as z:
    for m in z.namelist():
        if m.lower().endswith(".jpg"):
            (out / Path(m).name).write_bytes(z.read(m))
            n += 1
zip_path.unlink()
print(f"Listo: {n} imágenes en {out}")
