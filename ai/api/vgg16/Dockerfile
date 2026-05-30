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
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

COPY requirements/ /tmp/requirements/

RUN --mount=type=cache,target=/pip-cache \
    pip install --cache-dir=/pip-cache \
    -r /tmp/requirements/req_tensorflow.txt

RUN --mount=type=cache,target=/pip-cache \
    pip install --cache-dir=/pip-cache \
    -r /tmp/requirements/req_vision.txt

RUN --mount=type=cache,target=/pip-cache \
    pip install --cache-dir=/pip-cache \
    -r /tmp/requirements/req_misc.txt

RUN --mount=type=cache,target=/pip-cache \
    pip install --cache-dir=/pip-cache \
    -r /tmp/requirements/req_fastapi.txt

WORKDIR /app

RUN mkdir -p /app/vids /app/models /app/output

COPY pipeline_vgg16.py /app/pipeline_vgg16.py

COPY models/vgg16_unet_cornea.h5 /app/models/vgg16_unet_cornea.h5

CMD ["uvicorn", "pipeline_vgg16:app", "--host", "0.0.0.0", "--port", "8000"]