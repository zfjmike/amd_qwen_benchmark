# Qwen3-VL AMD Benchmark

Side-by-side performance comparison of **SGLang** vs **vLLM** serving Qwen3-VL on AMD GPUs.

## Requirements

- AMD GPU with ROCm support
- Docker
- Python 3.10+ with `requests` installed (`pip install requests`)
- A HuggingFace account / token if the model requires gated access

## Step 1 — Build Docker images

```bash
docker build -f Dockerfile.qwen3vl-rocm-sglang -t qwen3vl-sglang .
docker build -f Dockerfile.qwen3vl-rocm-vllm   -t qwen3vl-vllm .
```

## Step 2 — Add benchmark images

Place at least one image in the `images/` folder:

```bash
mkdir -p images
cp /path/to/your/image.jpg images/
```

## Step 3 — Launch the servers

Open two terminals and start each server. Both use GPU 0 by default; use the `GPU=` env var to assign different GPUs.

**Terminal 1 — SGLang** (port 30000):

```bash
GPU=0 PORT=30000 MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct bash qwen3vl_amd_sglang.sh
```

**Terminal 2 — vLLM** (port 8000):

```bash
GPU=1 PORT=8000 MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct bash qwen3vl_amd_vllm.sh
```

Wait until both servers print a ready message before proceeding. You can verify they are up:

```bash
curl http://localhost:30000/health   # SGLang
curl http://localhost:8000/health    # vLLM
```

### Optional server settings

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID or local path |
| `GPU` | `0` | GPU index to use |
| `PORT` | `30000` / `8000` | Listening port |
| `SHM_SIZE` | `16g` | Shared memory for the container |
| `HF_TOKEN` | _(empty)_ | HuggingFace token (vLLM only) |

Example — run on different GPUs with a smaller model:

```bash
GPU=0 MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct bash qwen3vl_amd_sglang.sh
GPU=1 MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct bash qwen3vl_amd_vllm.sh
```

## Step 4 — Run the benchmark

```bash
python benchmark_compare.py
```

This loads all images from `./images`, sends them to both servers concurrently, and prints a comparison table.

### Common options

```bash
# Limit to 20 images, increase concurrency
python benchmark_compare.py --max-examples 20 --concurrent-requests 64

# Use a custom image directory
python benchmark_compare.py --image-dir /path/to/images

# Benchmark only one server
python benchmark_compare.py --skip-vllm    # SGLang only
python benchmark_compare.py --skip-sglang  # vLLM only

# Custom server addresses or models
python benchmark_compare.py \
    --sglang-url http://localhost:30000 \
    --vllm-url   http://localhost:8000 \
    --sglang-model Qwen/Qwen3-VL-8B-Instruct \
    --vllm-model   Qwen/Qwen3-VL-8B-Instruct
```

### Full option reference

| Flag | Default | Description |
|---|---|---|
| `--image-dir` | `./images` | Directory of JPEG/PNG images |
| `--max-examples` | `-1` (all) | Max images to load |
| `--max-tokens` | `1024` | Max output tokens per request |
| `--warmup` | `2` | Warmup requests before measurement |
| `--concurrent-requests` | `32` | Number of parallel requests |
| `--request-timeout` | `600` | Per-request timeout in seconds |
| `--sglang-url` | `http://localhost:30000` | SGLang server URL |
| `--vllm-url` | `http://localhost:8000` | vLLM server URL |
| `--sglang-model` | `Qwen/Qwen3-VL-8B-Instruct` | Model name for SGLang |
| `--vllm-model` | `Qwen/Qwen3-VL-8B-Instruct` | Model name for vLLM |
| `--skip-sglang` | — | Skip SGLang server |
| `--skip-vllm` | — | Skip vLLM server |

## Notes on caching

Both SGLang and vLLM enable KV/prefix caching by default. Sending the same image repeatedly will hit the cache after the first request, resulting in lower measured latency. This reflects cached throughput — realistic for production workloads but not cold-start performance. Use diverse images for cold-cache measurements.


## SGLang bench_serving results comparison

Follow official documentation from SGLang and vLLM to build dockers. Then use sglang.bench_serving to collect perf metrics.

### Run SGLang

```bash
alias drun='docker run -it --rm --network=host --privileged --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $HOME/dockerx:/dockerx \
  -v /data:/data'
export SGLANG_USE_AITER=1

drun -p 30000:30000 \
  -e HIP_VISIBLE_DEVICES=6 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  sglang_image \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000

drun \
  sglang_image \
  bash -c "PYTHONPATH=/sgl-workspace/sglang/python python3 -m sglang.bench_serving \
  --backend sglang \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name image \
  --num-prompts 100 \
  --random-input-len 256 \
  --random-output-len 256 \
  --random-range-ratio 0.0 \
  --image-count 1 \
  --image-resolution 720p \
  --image-content random \
  --request-rate inf \
  --warmup-requests 1"
```

**Result**

```
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 not set
Successful requests:                     100
Benchmark duration (s):                  3.41
Total input tokens:                      102632
Total input text tokens:                 14432
Total input vision tokens:               88200
Total generated tokens:                  12508
Total generated tokens (retokenized):    11668
Request throughput (req/s):              29.36
Input token throughput (tok/s):          30128.48
Output token throughput (tok/s):         3671.83
Peak output token throughput (tok/s):    5062.00
Peak concurrent requests:                100
Total token throughput (tok/s):          33800.31
Concurrency:                             61.41
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   2091.83
Median E2E Latency (ms):                 2224.06
P90 E2E Latency (ms):                    3127.29
P99 E2E Latency (ms):                    3329.49
---------------Time to First Token----------------
Mean TTFT (ms):                          387.72
Median TTFT (ms):                        380.92
P99 TTFT (ms):                           500.25
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.85
Median TPOT (ms):                        14.77
P99 TPOT (ms):                           37.04
---------------Inter-Token Latency----------------
Mean ITL (ms):                           13.76
Median ITL (ms):                         14.20
P95 ITL (ms):                            17.33
P99 ITL (ms):                            18.34
Max ITL (ms):                            177.76
==================================================
```

### Run vLLM

```bash
alias drun='docker run -it --rm --network=host --privileged --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $HOME/dockerx:/dockerx \
    -v /data:/data'

drun --entrypoint python -p 8000:8000 \
    -e HIP_VISIBLE_DEVICES=2 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm-rocm \
    -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000

drun \
  sglang_image \
  bash -c "PYTHONPATH=/sgl-workspace/sglang/python python3 -m sglang.bench_serving \
  --backend vllm \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name image \
  --num-prompts 100 \
  --random-input-len 256 \
  --random-output-len 256 \
  --random-range-ratio 0.0 \
  --image-count 1 \
  --image-resolution 720p \
  --image-content random \
  --request-rate inf \
  --warmup-requests 1"
```

**Results**

```
============ Serving Benchmark Result ============
Backend:                                 vllm
Traffic request rate:                    inf
Max request concurrency:                 not set
Successful requests:                     100
Benchmark duration (s):                  2.20
Total input tokens:                      102620
Total input text tokens:                 14420
Total input vision tokens:               88200
Total generated tokens:                  12508
Total generated tokens (retokenized):    10531
Request throughput (req/s):              45.36
Input token throughput (tok/s):          46548.35
Output token throughput (tok/s):         5673.62
Peak output token throughput (tok/s):    6007.00
Peak concurrent requests:                100
Total token throughput (tok/s):          52221.97
Concurrency:                             59.61
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1314.12
Median E2E Latency (ms):                 1320.42
P90 E2E Latency (ms):                    1963.80
P99 E2E Latency (ms):                    2139.53
---------------Time to First Token----------------
Mean TTFT (ms):                          386.51
Median TTFT (ms):                        385.07
P99 TTFT (ms):                           482.34
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.25
Median TPOT (ms):                        7.44
P99 TPOT (ms):                           22.62
---------------Inter-Token Latency----------------
Mean ITL (ms):                           8.29
Median ITL (ms):                         7.36
P95 ITL (ms):                            14.84
P99 ITL (ms):                            15.52
Max ITL (ms):                            176.67
==================================================
```