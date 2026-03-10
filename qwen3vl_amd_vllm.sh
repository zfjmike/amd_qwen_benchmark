#!/usr/bin/env bash
# Launch the Qwen3-VL vLLM server on a single AMD GPU.
#
# Usage:
#   bash qwen3vl_amd_vllm.sh                                                # default: Qwen3-VL-8B-Instruct, GPU 0
#   GPU=2 bash qwen3vl_amd_vllm.sh                                          # use GPU 2
#   MODEL_PATH=Qwen/Qwen3-VL-2B-Instruct bash qwen3vl_amd_vllm.sh
#   bash qwen3vl_amd_vllm.sh --max-model-len 4096                           # pass extra vllm flags
#
# Build the image first:
#   docker build -f Dockerfile.qwen3vl-rocm-vllm -t qwen3vl-vllm .
#
# Default port is 8000 (vLLM convention) to avoid clashing with an SGLang
# server that may be running on 30000.

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
HF_TOKEN=${HF_TOKEN:-}
PORT=${PORT:-8000}
SHM_SIZE=${SHM_SIZE:-16g}
IMAGE=${IMAGE:-qwen3vl-vllm}
GPU=${GPU:-0}  # which GPU index to use (0-indexed)

docker run --rm -it \
  --network host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc host \
  --shm-size "$SHM_SIZE" \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
  --env MODEL_PATH="$MODEL_PATH" \
  --env PORT="$PORT" \
  --env HIP_VISIBLE_DEVICES="$GPU" \
  "$IMAGE" \
  "$@"
