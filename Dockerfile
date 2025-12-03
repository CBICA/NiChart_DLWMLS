
ARG CUDA_VERSION="12.1"
ARG TORCH_VERSION="2.3.1"
ARG CUDNN_VERSION="8"

## This base image is generally the smallest with all prereqs.
FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/CBICA/DLWMLS.git /DLWMLS
RUN pip install -e /DLWMLS
RUN DLWMLS -i /dummyinput -o /dummyoutput -device cpu

WORKDIR /app
COPY . /app/ 

RUN pip install -e .

ENV MKL_SERVICE_FORCE_INTEL=1
ENTRYPOINT ["NiChart_DLWMLS"]

