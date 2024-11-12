FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git clang build-essential wget

# Add NVIDIA package repositories for additional libraries
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y libnvidia-ml-dev openmpi-bin libopenmpi-dev curl

RUN git clone https://github.com/NVIDIA/cudnn-frontend \
 && git clone https://github.com/karpathy/llm.c \
 && mv cudnn-frontend llm.c /root

RUN cd ~/llm.c && chmod u+x ./dev/download_starter_pack.sh && ./dev/download_starter_pack.sh
RUN cd ~/llm.c && USE_CUDNN=1 make
