FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.128.0 \
    uvicorn==0.46.0 \
    transformers==5.7.0 \
    accelerate==1.13.0 \
    safetensors==0.7.0 \
    bitsandbytes==0.49.2 \
    httpx==0.28.1 \
    python-multipart \
    huggingface_hub==1.10.1 \
    tokenizers==0.22.2

COPY main.py /app/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]