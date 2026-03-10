#!/usr/bin/env bash
# Launch the Qwen3-VL SGLang server on a single AMD GPU.
#
# Usage:
#   bash qwen3vl_amd_sglang.sh                                              # default: Qwen3-VL-8B-Instruct, GPU 0
#   GPU=1 bash qwen3vl_amd_sglang.sh                                        # use GPU 1
#   MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct bash qwen3vl_amd_sglang.sh
#   bash qwen3vl_amd_sglang.sh --mem-fraction-static 0.90                   # pass extra sglang flags
#
# Build the image first:
#   docker build -f Dockerfile.qwen3vl-rocm-sglang -t qwen3vl-sglang .
#
# Example with logging:
#   GPU=1 MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct bash qwen3vl_amd_sglang.sh --mem-fraction-static 0.85 2>&1 | tee sglang_server.log
#
# The HuggingFace model cache is mounted from ~/.cache/huggingface so
# models are only downloaded once.

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
PORT=${PORT:-30000}
SHM_SIZE=${SHM_SIZE:-16g}
IMAGE=${IMAGE:-qwen3vl-sglang}
GPU=${GPU:-0}  # which GPU index to use (0-indexed)

docker run --rm -it \
  --privileged \
  --network host \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --ipc host \
  --shm-size "$SHM_SIZE" \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $HOME/dockerx:/root/dockerx \
  -v /data:/data \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/aiter:/root/.aiter" \
  -v "$HOME/.cache/aiter-jit:/sgl-workspace/aiter/aiter/jit/build" \
  --env MODEL_PATH="$MODEL_PATH" \
  --env PORT="$PORT" \
  --env ROCR_VISIBLE_DEVICES="$GPU" \
  "$IMAGE" \
  "$@"
