FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
RUN apt update
RUN apt install -y git

RUN apt install -y software-properties-common python-software-properties
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y python3.6 python3.6-dev python3-pip
RUN apt install -y libgl1-mesa-glx libsm6 

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 2
RUN update-alternatives --set python3 /usr/bin/python3.6

#RUN git clone https://github.com/HimangiM/Self-Supervised-Scene-Flow-Estimation.git flow
COPY requirements.txt requirements.txt
RUN pip3 --version
RUN pip3 install --upgrade "pip < 21.0"
RUN python3 --version
RUN pip3 install -r requirements.txt 

ENV CUDA_HOME /usr/local/cuda
ENV CUDA_ARCH sm_86

COPY . flow

WORKDIR /flow/src/tf_ops/3d_interpolation
RUN make

WORKDIR /flow/src/tf_ops/grouping
RUN make

WORKDIR /flow/src/tf_ops/sampling
RUN make

WORKDIR /flow
