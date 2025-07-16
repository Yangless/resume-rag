FROM nvidia/cuda:12.4.0-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
