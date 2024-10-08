FROM intel/oneapi

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
USER root

RUN apt  update  -y --allow-insecure-repositories

RUN apt install -y  build-essential git cmake  wget gpg \
					python3-pip libopencv-dev python3-opencv libqt5widgets5 

					
WORKDIR /workspace/

COPY ./requirements.txt  /workspace
RUN	pip3 install -r ./requirements.txt

RUN	wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
RUN	echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal main' | tee -a /etc/apt/sources.list.d/intel.list
RUN	apt update -y --allow-insecure-repositories
RUN	apt install -y  ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero
RUN	pip3 install dpctl dpnp openvino-dev[onnx]
RUN	pip3 install torch==2.3.1 torchvision==0.18.1 intel_extension_for_pytorch==2.3.110+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
	
	


