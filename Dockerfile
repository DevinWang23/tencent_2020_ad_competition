# py3, cuda10.1, pytorch1.4 (GPU) and spark2.4.5
FROM registry.cn-shenzhen.aliyuncs.com/mengqiu/machine_learning:pytorch1.4-cuda10.1-py3-spark2.4.5

MAINTAINER MengQiu Wang "wangmengqiu@ainnovation.com"

ENV REFRESHED_AT 2020-03-28
ENV PYTHONIOENCODING=UTF-8
ENV LANG C.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

USER root

