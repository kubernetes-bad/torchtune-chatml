FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

WORKDIR /app/training

RUN apt update && apt install -y nano screen python3-pip build-essential cmake git
RUN pip install --pre torchtune bitsandbytes --extra-index-url https://download.pytorch.org/whl/nightly/cu121 --no-cache-dir
