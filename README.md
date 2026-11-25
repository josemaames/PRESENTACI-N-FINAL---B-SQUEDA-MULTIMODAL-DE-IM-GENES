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
- Selector de método CLIP | SigLIP para comparar dos modelos sobre las mismas imágenes (ver más abajo).

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
python -m app.main --build-siglip # opcional: índice del método 2 (SigLIP, ~40 min en CPU)
python -m uvicorn app.main:app --port 8000
```

Luego abre http://localhost:8000. La primera ejecución descarga los modelos (~1,1 GB en total).
Cualquier carpeta con 1000 o más imágenes en `./images` sirve en lugar de los datasets de ejemplo.

## Comparación de dos métodos: CLIP frente a SigLIP
Además del método principal (CLIP), hay un segundo método para compararlos sobre las mismas 12 390 imágenes.
En la interfaz, el selector **Método: CLIP | SigLIP** cambia de uno a otro y repite la última búsqueda
(texto, imagen o "Ver similares"); cada resultado indica con qué método se obtuvo.

- **Método 1, CLIP ViT-B/32** (OpenAI): vectores de 512 dimensiones. Para el texto se usa la versión multilingüe
  (`clip-ViT-B-32-multilingual-v1`), que permite buscar en español.
- **Método 2, SigLIP ViT-B/16** (Google, `siglip-base-patch16-256-multilingual`): vectores de 768 dimensiones. También
  relaciona texto e imagen, pero se entrenó con otra función de pérdida (sigmoide en lugar de softmax contrastivo) y su
  texto multilingüe es nativo, no un modelo aparte.

Se construye su índice una vez (unos 40 min en CPU; guarda el avance y se puede continuar si se interrumpe):

```bash
python -m app.main --build-siglip   # crea index/embeddings_siglip.npy
python compare_methods.py           # mide la comparación y genera docs/comparacion_*
```

**Cómo se comparó.** `compare_methods.py` usa búsqueda exacta en ambos casos, así que la única diferencia es el modelo que
genera los vectores. Como solo las mascotas tienen etiqueta (la raza), la precisión se mide con ellas; COCO se compara
con acuerdo entre métodos y con la figura `docs/comparacion_cualitativa.png`.

| Métrica (top 6, salvo indicación) | CLIP (texto multilingüe, el de la app) | CLIP (texto original) | SigLIP |
| --- | --- | --- | --- |
| Imagen → imagen, misma raza (1 500 consultas) | 72,3 % | 72,3 % | **88,6 %** |
| Texto → imagen, raza correcta @6 (37 consultas) | 13,5 % | 90,5 % | **97,3 %** |
| Texto → imagen, raza correcta @20 | 12,6 % | 89,2 % | **96,5 %** |
| "Un gato" / "un perro", especie correcta @20, español | 100 % | no aplica | 97,5 % |
| "A cat" / "a dog", especie correcta @20, inglés | 100 % | 100 % | 92,5 % |

| Costo (CPU) | CLIP | SigLIP |
| --- | --- | --- |
| Dimensiones del vector / tamaño del índice | 512 / 25,4 MB | 768 / 38,1 MB |
| Codificar una imagen | 37 ms | 171 ms |
| Codificar una consulta de texto | 12 ms | 51 ms |
| Buscar entre 12 390 vectores (exacta) | 0,96 ms | 1,11 ms |

**Qué se observa**
- SigLIP distingue mucho mejor las razas, tanto con texto como con imagen. Es el mejor método cuando importa el detalle fino.
- CLIP cuesta unas 4,6 veces menos al codificar una imagen y su índice es más pequeño; la búsqueda en sí tarda lo mismo.
- En texto → raza, CLIP cae a 13,5 % por el codificador multilingüe que usa la app, no por CLIP en sí: con su texto original
  llega a 90,5 %. Ese codificador es una versión comprimida que prioriza entender 50 idiomas y pierde precisión en nombres
  de razas. Para especies y escenas generales (gato, perro, pizza, caballo) funciona bien.
- Los dos métodos no son intercambiables: comparten solo un 20 % de los 6 resultados en texto y un 34 % con imágenes.
  Ambos devuelven resultados razonables, pero distintos (ver `docs/comparacion_cualitativa.png`).
- Los puntajes de similitud no se pueden comparar entre métodos: CLIP da valores cercanos a 0,30 y SigLIP cercanos a 0,12.

**Limitaciones de la comparación.** La precisión por raza solo cubre mascotas (37 clases, una plantilla de consulta
"a photo of a …"); COCO se evalúa solo de forma cualitativa y por acuerdo. La prueba de especie cuenta como error cualquier
foto de COCO, aunque muestre un perro o un gato, por lo que subestima a los métodos que devuelven fotos de COCO (no se verificó
cuántos de los errores de SigLIP son de ese tipo). Los tiempos son de una sola máquina, en CPU. SigLIP siempre busca en
memoria; con `SEARCH_BACKEND=postgres` solo CLIP usa PostgreSQL.

## Base de datos: PostgreSQL + pgvector (opcional)
Por defecto el índice vive en memoria (`index/embeddings.npy` + FAISS). Para guardar imágenes, metadatos y
vectores en una base de datos real, hay un modo con **PostgreSQL + pgvector** (necesita Docker):

```bash
docker compose up -d              # PostgreSQL con pgvector en el puerto 5433
python migrate_to_db.py --reset   # carga los embeddings y crea el índice HNSW
```

Luego se arranca el servidor eligiendo el backend (`faiss` por defecto, o `postgres`):

```bash
# PowerShell
$env:SEARCH_BACKEND="postgres"; python -m uvicorn app.main:app --port 8000
```

Tabla `images`: `id` · `path` (única) · `dataset` (`pets` o `coco`) · `label` (raza, en mascotas) · `embedding vector(512)`.
La búsqueda es una consulta SQL ordenada por distancia coseno (`ORDER BY embedding <=> $1 LIMIT 6`) que usa un índice HNSW.
La conexión se configura con `DATABASE_URL` (por defecto `postgresql://imagesearch:imagesearch@localhost:5433/imagesearch`,
credenciales de desarrollo local; cámbialas si lo expones fuera de tu equipo).

Con las 12 390 imágenes, pgvector coincide con la búsqueda exacta en el 99,8 % de los resultados (recall@6 = 0,998).
`benchmark_pg.py` mide el comportamiento a gran escala (usa una tabla temporal, no toca `images`). Resultados en CPU,
con Docker, 105 consultas, top 6, y vectores adicionales sintéticos creados a partir de embeddings reales:

| Vectores | Exacta (sin índice) | HNSW `ef_search=64` | Recall@6 | Construcción del índice | Tamaño del índice |
| --- | --- | --- | --- | --- | --- |
| 100 000 | 429 ms | 3,9 ms | 0,998 | 90 s | 273 MB |
| 500 000 | 1 144 ms | 6,6 ms | 0,997 | 215 s | 1 365 MB |

Con `ef_search=40` baja a 4,0 ms (recall 0,994) y con `ef_search=128` sube a 9,9 ms (recall 0,998). En comparación con FAISS HNSW
en memoria (0,87 ms con 500 000 vectores), PostgreSQL es más lento por consulta, a cambio de persistencia, metadatos y filtros SQL.
Los tiempos de PostgreSQL incluyen el viaje de ida y vuelta al contenedor.

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
- `compare_methods.py`: comparación CLIP frente a SigLIP · `docs/comparacion_resultados.json` y `docs/comparacion_cualitativa.png`: resultados
- `download_dataset.py`, `download_coco.py`, `make_deploy.py`, `benchmark.py`
- `deploy/`: Dockerfile y archivos para un Space de Hugging Face (requiere plan de pago para Docker)
- `presentacion.pptx`: presentación final
- `docs/presentacion.pdf` y `docs/informe.pdf`: presentación e informe en PDF
- `docker-compose.yml`, `app/db.py`, `migrate_to_db.py`, `benchmark_pg.py`: modo PostgreSQL + pgvector

## Limitaciones
- CLIP es débil contando objetos ("two dogs") y el sistema solo conoce lo que hay en el dataset.
- No se midió precisión con respuestas esperadas; el recall compara contra la búsqueda exacta.
- Los tiempos de codificación de imágenes subidas se midieron en CPU (~750 ms).
