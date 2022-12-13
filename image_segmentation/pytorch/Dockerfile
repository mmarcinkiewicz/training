#ARG FROM_IMAGE_NAME=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ARG FROM_IMAGE_NAME=gitlab-master.nvidia.com/dl/dgx/pytorch:master-py3-devel
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/unet3d
WORKDIR /workspace/unet3d

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install -y vim

RUN pip install --upgrade pip
RUN pip install --disable-pip-version-check -r requirements.txt

RUN pip install monai==1.0.0 --no-dependencies
RUN pip install torchmetrics --no-dependencies

#RUN pip uninstall -y apex; pip uninstall -y apex; git clone --branch seryilmaz/fused_dropout_softmax  https://github.com/seryilmaz/apex.git; cd apex;  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--xentropy" --global-option="--deprecated_fused_adam" --global-option="--deprecated_fused_lamb" --global-option="--fast_multihead_attn" .
