---
title: Buscador de Imágenes CLIP
emoji: 🔎
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Buscador multimodal de imágenes (CLIP)

Búsqueda por texto (español e inglés) y por imagen sobre 12 390 fotos
(Oxford-IIIT Pet + COCO val2017, reducidas a 512 px para este despliegue).

- Imágenes: `openai/clip-vit-base-patch32`
- Texto: `sentence-transformers/clip-ViT-B-32-multilingual-v1`
- Backend: FastAPI · Índice: FAISS

Proyecto final de Tópicos avanzados en Machine Learning.
