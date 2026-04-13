FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .



RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt
    
RUN --mount=type=cache,target=/root/.cache/torch
RUN python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)"

COPY app ./app
COPY ml ./ml
COPY .env .env

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
