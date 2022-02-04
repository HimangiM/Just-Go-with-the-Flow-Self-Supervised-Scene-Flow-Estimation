FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
ENV DEBIAN_FRONTEND=noninteractive \
  CUDA_HOME=/usr/local/cuda \
  CUDA_ARCH=sm_86 \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
  PATH=/usr/local/cuda/bin:$PATH

RUN apt update && \
  apt install -y git software-properties-common zlib1g-dev

RUN apt-get install -y build-essential checkinstall libreadline-gplv2-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
RUN git clone https://github.com/deadsnakes/python3.6.git /opt/python3.6 && \
  cd /opt/python3.6 && \
  git checkout ubuntu/xenial && \
  ./configure --enable-optimizations && \
  make && \
  make install

RUN apt install -y python3-pip libgl1-mesa-glx libsm6
#  add-apt-repository -y ppa:deadsnakes/ppa && \
#  apt update && \
#  apt install -y python3.6 python3.6-dev python3-pip && \
#  apt install -y libgl1-mesa-glx libsm6 
#
#RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
#  update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 && \
#  update-alternatives --set python3 /usr/bin/python3.6 && \
#  update-alternatives --set python /usr/bin/python3.6
RUN pip3 install --upgrade "pip < 21.0"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt 


COPY . flow

RUN cd /flow/src/tf_ops/3d_interpolation && make && \
  cd /flow/src/tf_ops/grouping && make && \
  cd /flow/src/tf_ops/sampling && make

WORKDIR /flow
RUN mkdir -p data_preprocessing/kitti_self_supervised_flow && \
  mkdir -p log_train_pretrained && \
  chmod +x src/commands/command_evaluate_kitti.sh
#CMD src/commands/command_evaluate_kitti.sh
