# Buscador multimodal de imágenes (CLIP)

Búsqueda texto→imagen e imagen→imagen sobre cualquier carpeta de imágenes (≥1000).

## Uso

```bash
pip install -r requirements.txt
python download_dataset.py        # opcional: Oxford-IIIT Pet (~3700 imgs). O copia tus imágenes a ./images
python -m app.main --build        # extrae embeddings con CLIP y crea ./index
uvicorn app.main:app --port 8000  # abre http://localhost:8000
```

## Cómo funciona
- `openai/clip-vit-base-patch32` (Hugging Face) mapea texto e imágenes al mismo espacio de 512 dims.
- Los embeddings se normalizan y se guardan en `index/embeddings.npy`.
- La búsqueda es producto interno (= similitud coseno) con FAISS (`IndexFlatIP`), o numpy si no está instalado.
- Escalabilidad: para millones de imágenes, cambiar a `IndexIVFFlat`/`HNSW` en FAISS; el indexado por lotes ya está soportado.
