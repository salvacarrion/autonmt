# syntax=docker/dockerfile:1.6
#
# Dev/training image for AutoNMT. Defaults to the latest published PyTorch
# image; pin via build-arg for reproducible builds, e.g.:
#
#   docker build --build-arg PYTORCH_TAG=2.4.1-cuda12.1-cudnn9-runtime -t autonmt .
#
# Browse available tags at: https://hub.docker.com/r/pytorch/pytorch/tags
ARG PYTORCH_TAG=latest
FROM pytorch/pytorch:${PYTORCH_TAG}

# --- system packages -----------------------------------------------------
# Single RUN keeps the layer-cache coherent: editing the package list
# invalidates the apt update too. --no-install-recommends + cleanup of
# /var/lib/apt/lists keeps the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        htop \
        tmux \
        vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/autonmt

# --- python deps (cached layer) ------------------------------------------
# Copy only the install metadata first so changes to source code don't
# invalidate the pip cache. Heavy deps (torch/lightning/sentencepiece/...)
# get installed once and reused on every rebuild that doesn't touch
# requirements.txt or setup.py.
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- source --------------------------------------------------------------
COPY . .
RUN pip install --no-cache-dir -e .

# Keep the container alive so developers can `docker exec` into it.
CMD ["tail", "-f", "/dev/null"]
