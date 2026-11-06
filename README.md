# Buscador multimodal de imágenes con CLIP

Proyecto final de **Tópicos avanzados en Machine Learning**. Plataforma web que encuentra
imágenes por **texto** (español e inglés) o por **similitud con otra imagen**, sobre
12 390 fotos (Oxford-IIIT Pet + COCO val2017).

## Funciones
- Búsqueda por texto en español e inglés y por imagen (subir, arrastrar, pegar con Ctrl+V o tomar foto).
- Búsqueda por voz, historial de consultas y botón "Ver similares" en cada resultado.
- Tiempo de cada consulta, puntaje de similitud, copiar enlace y descargar.
- Modo claro y modo noche, imagen ampliada con navegación y atajos de teclado.
- Modal "¿Cómo funciona?" que explica la idea de los vectores.

## Cómo funciona
- **Imágenes:** `openai/clip-vit-base-patch32` (Hugging Face) las convierte en vectores de 512 dimensiones.
- **Texto:** `sentence-transformers/clip-ViT-B-32-multilingual-v1` proyecta texto en español, inglés y otros idiomas
  al mismo espacio, por lo que el índice de imágenes no cambia.
- Los vectores se normalizan y se guardan en `index/embeddings.npy`. La búsqueda usa producto interno
  (equivale a similitud coseno) con FAISS (`IndexFlatIP`) o numpy.
- Backend: FastAPI. Interfaz: HTML, CSS y JavaScript sin librerías.

## Ejecutar
Los datasets y el índice no están en el repositorio (pesan más de 1,5 GB); se generan con estos pasos:

```bash
pip install -r requirements.txt
python download_dataset.py        # Oxford-IIIT Pet (7 390 imágenes)
python download_coco.py           # COCO val2017 (5 000 imágenes)
python -m app.main --build        # extrae embeddings y crea ./index
python -m uvicorn app.main:app --port 8000
```

Luego abre http://localhost:8000. La primera ejecución descarga los modelos (~1,1 GB en total).
Cualquier carpeta con 1000 o más imágenes en `./images` sirve en lugar de los datasets de ejemplo.

## Escalabilidad
`benchmark.py` compara búsqueda exacta (numpy y FAISS Flat) con índices aproximados (IVF y HNSW).
Tiempo por consulta en CPU, con vectores adicionales sintéticos creados a partir de embeddings reales:

| Vectores | Exacta (numpy) | IVF | HNSW (recall@6) |
| --- | --- | --- | --- |
| 7 390 | 0,34 ms | 0,14 ms | 0,19 ms (0,99) |
| 100 000 | 8,61 ms | 2,97 ms | 0,34 ms (0,98) |
| 500 000 | 49,94 ms | 28,60 ms | 0,87 ms (0,97) |
| 1 000 000 | 90,10 ms | 69,12 ms | 1,13 ms (0,96) |

HNSW es unas 80 veces más rápido que la búsqueda exacta con un millón de vectores, a cambio de
más tiempo de construcción (295 s) y más memoria. Se ejecuta con `python benchmark.py --big`.

## Archivos
- `app/main.py`: API y búsqueda · `static/index.html`: interfaz
- `download_dataset.py`, `download_coco.py`, `make_deploy.py`, `benchmark.py`
- `deploy/`: Dockerfile y archivos para un Space de Hugging Face (requiere plan de pago para Docker)
- `presentacion.pptx`: presentación final

## Limitaciones
- CLIP es débil contando objetos ("two dogs") y el sistema solo conoce lo que hay en el dataset.
- No se midió precisión con respuestas esperadas; el recall compara contra la búsqueda exacta.
- Los tiempos de codificación de imágenes subidas se midieron en CPU (~750 ms).
