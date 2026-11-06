
ARG CUDA_VERSION="12.1"
ARG TORCH_VERSION="2.3.1"
ARG CUDNN_VERSION="8"

## This base image is generally the smallest with all prereqs.
FROM pytorch/pytorch:${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/CBICA/DLWMLS.git /DLWMLS
RUN pip install -e /DLWMLS

WORKDIR /app
COPY . /app/ 

RUN pip install -e .

# Bake the nnU-Net weights into the image. dlwmls_pipeline.py downloads them
# into this same directory at runtime whenever it finds them missing, which
# makes every run depend on reaching Hugging Face and fails outright when the
# container runs as a non-root user, since it cannot write /DLWMLS. The previous
# attempt at this, `RUN DLWMLS -i /dummyinput -o /dummyoutput -device cpu`,
# exits 0 without downloading anything, so the image shipped without them.
# The rm clears the hub cache in case anything else has populated it. The test
# is what stops this failing silently again.
RUN python scripts/fetch_model.py \
 && rm -rf /root/.cache/huggingface /DLWMLS/DLWMLS/.cache \
 && test -f "/DLWMLS/DLWMLS/nnunet_results/Dataset004_Dataset003_qcedCorrected/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth" \
 && chmod -R a+rX /DLWMLS/DLWMLS/nnunet_results

ENV MKL_SERVICE_FORCE_INTEL=1
ENTRYPOINT ["NiChart_DLWMLS"]

