FROM python:3.11-slim

# Hugging Face Spaces ejecuta el contenedor como usuario 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# PyTorch solo CPU (mucho más liviano que la versión con CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descarga los modelos al construir la imagen para que el arranque sea rápido
USER user
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')"

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
