FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_CACHE_DIR=/pip-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

COPY requirements/ /tmp/requirements/

RUN --mount=type=cache,target=/pip-cache \
    pip install --cache-dir=/pip-cache \
    -f https://download.pytorch.org/whl/cu121/torch_stable.html \
    -r /tmp/requirements/req_torch.txt

RUN --mount=type=cache,target=/pip-cache \
    pip install --cache-dir=/pip-cache \
    -r /tmp/requirements/req_transformers.txt

RUN --mount=type=cache,target=/pip-cache \
    pip install --cache-dir=/pip-cache \
    -r /tmp/requirements/req_vision.txt

RUN --mount=type=cache,target=/pip-cache \
    pip install --cache-dir=/pip-cache \
    -r /tmp/requirements/req_misc.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/req_fastapi.txt /tmp/requirements/req_fastapi.txt
RUN --mount=type=cache,target=/pip-cache \
    pip install --cache-dir=/pip-cache \
    -r /tmp/requirements/req_fastapi.txt


WORKDIR /app
COPY pipeline_segformer.py /app/

RUN mkdir -p /app/vids /app/models /app/output

COPY models/best_segformer_b0_lightbar /app/models/best_segformer_b0_lightbar

COPY pipeline_segformer.py /app/pipeline_segformer.py

CMD ["uvicorn", "pipeline_segformer:app", "--host", "0.0.0.0", "--port", "8000"]
