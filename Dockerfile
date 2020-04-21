# py3, cuda10.1 and pytorch1.4 (GPU)

FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3

MAINTAINER MengQiu Wang "wangmengqiu@ainnovation.com"

ENV REFRESHED_AT 2020-03-28
ENV PYTHONIOENCODING=UTF-8
ENV LANG C.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

USER root

# install packages
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends\
    software-properties-common \
    gcc \
	g++\
	wget\
	make \
	zlib1g \
	zlib1g-dev \
	build-essential \
	openssl \
	curl \
	libssl-dev \
	libffi-dev \
	vim \
	libbz2-dev \
	python3-tk \
	tk-dev \
	liblzma-dev \
	libsm6 \
	libxext6 \
    libxrender-dev \
    tmux \
	zip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# set up work directory
WORKDIR mlpipeline
COPY . .
#COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com