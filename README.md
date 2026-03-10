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
