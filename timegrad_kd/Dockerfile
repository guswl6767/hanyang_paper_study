# base-conda
# FROM continuumio/miniconda3 AS conda_base
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu20.04 AS cuda_base

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME="/root"
ENV PYTHON_VERSION=3.9
ENV PROJECT_PATH="${HOME}/Projects/hanyang_paper_study"


WORKDIR ${PROJECT_PATH}

# 필요한 패키지 설치
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1-mesa-glx \
       libglib2.0-0 \
       wget \
       curl \
       make \
       build-essential \
       libssl-dev \
       zlib1g-dev \
       libbz2-dev \
       libreadline-dev \
       libsqlite3-dev \
       libncurses5-dev \
       xz-utils \
       tk-dev \
       libxml2-dev \
       libxmlsec1-dev \
       libffi-dev \
       liblzma-dev \
       git \
       vim \
       tmux

# # Conda 환경 생성
# RUN conda create -n ${CONDA_DEFAULT_ENV} python=${PYTHON_VERSION} -y && conda clean -afy
# SHELL ["conda", "run", "-n", "jumprope_game", "/bin/bash", "-c"]

# COPY requirements.txt .
# RUN pip install -r requirements.txt

# # Conda 환경 활성화 설정
# RUN echo "conda activate ${CONDA_DEFAULT_ENV}" >> /root/.bashrc



# # 기본 명령어
# CMD ["bash"]
