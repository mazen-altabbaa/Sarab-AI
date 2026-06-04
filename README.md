# Sarab-AI — AI Module

This directory contains all AI-related components of the Sarab-AI project: trained models, training notebooks, and the three containerized inference services that power the system.

---

## Directory Structure

```
ai/
├── api/
│   ├── docker-compose.yml
│   ├── vgg16/
│   │   ├── Dockerfile
│   │   ├── models/
│   │   ├── pipeline_vgg16.py
│   │   └── requirements/
│   │       ├── req_fastapi.txt
│   │       ├── req_misc.txt
│   │       ├── req_tensorflow.txt
│   │       ├── req_torch.txt
│   │       ├── req_transformers.txt
│   │       └── req_vision.txt
│   ├── voicesystem/
│   │   ├── Dockerfile
│   │   └── main.py
│   └── whisper/
│       ├── Dockerfile
│       └── webservice.py
├── models/
│   └── vgg16_unet_cornea.h5
└── pipeline/
    ├── VGG16.ipynb
    ├── Qwen3_5_2B_Syrian_Finetune.ipynb
    ├── showSiriusMap.py
    └── transformData.py
```

---

## Services

### 1. VGG16 Cornea Segmentation (`api/vgg16`)

A FastAPI service that runs cornea segmentation inference using a custom-trained VGG16-UNet model.

**Base image:** `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`  
**Framework:** TensorFlow / Keras  
**Exposed port:** `8000`  
**Model:** `vgg16_unet_cornea.h5` (trained from scratch — see `pipeline/VGG16.ipynb`)

The model was trained using the notebook in `pipeline/VGG16.ipynb` and the resulting `.h5` weights file is stored in `models/` and copied into the container at build time.

Build and run:
```bash
cd api/vgg16
docker build -t vgg16-pipeline:v1 .
docker run --gpus all -p 8000:8000 vgg16-pipeline:v1
```

---

### 2. Voice System / LLM (`api/voicesystem`)

A FastAPI service that serves the fine-tuned Qwen 3.5 2B language model for conversational AI.

**Base image:** `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`  
**Framework:** Transformers (Hugging Face)  
**Exposed port:** `8000`  
**Model:** Qwen 3.5 2B, fine-tuned on a Syrian medical dataset (see `pipeline/Qwen3_5_2B_Syrian_Finetune.ipynb`)

The model was fine-tuned using the notebook in `pipeline/Qwen3_5_2B_Syrian_Finetune.ipynb` and downloaded from Hugging Face into the container at runtime.

Build and run:
```bash
cd api/voicesystem
docker build -t voicesystem-api:v1 .
docker run --gpus all -p 8000:8000 voicesystem-api:v1
```

---

### 3. Whisper ASR (`api/whisper`)

A speech-to-text service built on top of the `whisper-small-cpu` base image.

**Base image:** `whisper-small-cpu:latest` (pulled from Docker Hub: `onerahmet/openai-whisper-asr-webservice`)  
**Model:** Whisper small  
**Exposed port:** `9000`

The base image is pulled once from Docker Hub and the local Dockerfile simply copies in the webservice entrypoint and sets the model variant via environment variable.

Build and run:
```bash
cd api/whisper
docker build -t whisper-api:v1 .
docker run -p 9000:9000 -e ASR_MODEL=small whisper-api:v1
```

---

## Running All Services Together

A `docker-compose.yml` is provided in `api/` to bring up the full stack:

```bash
cd api
docker compose up
```

Services and their ports:

| Service     | Port  | Description                    |
|-------------|-------|--------------------------------|
| whisper     | 9000  | Speech-to-text (ASR)           |
| voicesystem | 8000  | LLM inference (Qwen fine-tune) |
| vgg16       | 10000 | Cornea segmentation            |

The `voicesystem` service depends on `whisper` being healthy before starting.

---

## Training Notebooks

Both models were trained locally and their notebooks are kept in `pipeline/` for reproducibility.

| Notebook | Description |
|---|---|
| `VGG16.ipynb` | Trains a VGG16-UNet architecture for cornea image segmentation. Outputs `vgg16_unet_cornea.h5`. |
| `Qwen3_5_2B_Syrian_Finetune.ipynb` | Fine-tunes the Qwen 3.5 2B base model on a Syrian medical dialogue dataset. Model is exported to Hugging Face Hub for use in the voicesystem container. |

---

## Requirements

- Docker with NVIDIA Container Toolkit (for GPU services)
- CUDA-compatible GPU (VGG16 and voicesystem services)
- The `whisper-small-cpu` base image must be pulled before building the whisper service:

```bash
docker pull onerahmet/openai-whisper-asr-webservice:latest
docker tag onerahmet/openai-whisper-asr-webservice:latest whisper-small-cpu:latest
```

## Client Application (React Native / Expo)
This section describes how to set up, run, and export the mobile application client (Sarab AI) for Android.

### 1. Prerequisites & Environment Setup
The application communicates with the backend via environment variables and utilizes Expo SDK with native modules (Camera, Video, Secure Store).

### 2. Local Development Installation
To set up local packages and sync dependencies across repository branches after synchronization or merging:

```bash
cd client
Install dependencies securely bypassing legacy peer conflicts
npm install --legacy-peer-deps
```
### 3. Run
```bash
npx expo start
```
