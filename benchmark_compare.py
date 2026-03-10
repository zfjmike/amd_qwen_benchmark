"""Side-by-side performance comparison of SGLang vs vLLM on local images.

Reads JPEG/PNG images from a local directory, sends each image + instruction
to both servers concurrently, and prints a comparison table of per-request latency
and overall throughput (QPS).

Usage:
  # Both servers running locally on their default ports
  python benchmark_compare.py --image-dir ./images

  # Custom server addresses
  python benchmark_compare.py \
      --sglang-url http://localhost:30000 \
      --vllm-url   http://localhost:8000 \
      --image-dir  ./images \
      --max-examples 10

  python benchmark_compare.py --image-dir ./images
"""

import argparse
import base64
import pathlib
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

# Same instruction used in batch inference scripts.
DEFAULT_INSTRUCTION = """
Describe the image in a json format such as:

{
    "high_level_description": "a high level description of the entire image",
    "style": ["style1", "style2", "etc"],
    "taxonomy_level1": "level 1 taxonomy",
    "taxonomy_level2": "level 2 taxonomy",
    "author_and_artwork_name": "author and or artwork name",
    "detailed_descriptions": ["description1", "description2", "etc"],
    "text_in_image": [
        {
            "text": "the text in the image, exactly as you see it",
            "style": "style of text",
            "color": "color of the text",
            "size": "the size of the text",
            "position": "position of the text in the image",
        },
        ...
    ],

}

Be as specific as possible.
* For example, if you recognize a famous character or person like Naruto or Donald Trump, mention the name.
* For example, if you recognize a specific car (like porsche 911), mention the specific name of the car.
* For example, if you recognize a specific location, video game, or any other proper noun, mention it.
"""


def _image_bytes_to_data_url(image_bytes: bytes) -> str:
  b64 = base64.b64encode(image_bytes).decode()
  return f"data:image/jpeg;base64,{b64}"


def _build_payload(model: str, image_bytes: bytes, instruction: str, max_tokens: int) -> dict[str, Any]:
  return {
    "model": model,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": _image_bytes_to_data_url(image_bytes)}},
          {"type": "text", "text": instruction},
        ],
      }
    ],
    "max_tokens": max_tokens,
  }


def _run_once(server_url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
  url = f"{server_url}/v1/chat/completions"
  t_start = time.perf_counter()
  resp = requests.post(url, json=payload, timeout=timeout)
  resp.raise_for_status()
  latency = time.perf_counter() - t_start
  data = resp.json()
  output_tokens = data.get("usage", {}).get("completion_tokens", 0)
  return {"latency_s": latency, "output_tokens": output_tokens}


def _load_images(image_dir: str, max_examples: int | None) -> list[bytes]:
  """Returns list of image bytes from a local directory."""
  exts = {".jpg", ".jpeg", ".png", ".webp"}
  paths = sorted(p for p in pathlib.Path(image_dir).iterdir() if p.suffix.lower() in exts)
  if max_examples is not None and max_examples >= 0:
    paths = paths[:max_examples]
  if not paths:
    raise ValueError(f"No images found in {image_dir!r} (looking for {exts})")
  return [p.read_bytes() for p in paths]


def _benchmark_server(
  label: str,
  server_url: str,
  model: str,
  images: list[bytes],
  instruction: str,
  max_tokens: int,
  concurrent_requests: int,
  request_timeout: int,
  warmup: int,
) -> dict[str, Any] | None:
  """Returns aggregated stats dict, or None if server is unreachable."""
  try:
    requests.get(f"{server_url}/health", timeout=10).raise_for_status()
  except Exception as e:
    print(f"  [{label}] unreachable: {e}")
    return None

  print(f"\n[{label}]  {server_url}")

  def _submit(img_bytes: bytes) -> dict[str, Any]:
    payload = _build_payload(model, img_bytes, instruction, max_tokens)
    return _run_once(server_url, payload, request_timeout)

  if warmup > 0:
    warmup_images = [images[i % len(images)] for i in range(warmup)]
    print(f"  Warming up ({warmup} requests)...")
    with ThreadPoolExecutor(max_workers=min(warmup, concurrent_requests)) as pool:
      for i, r in enumerate(pool.map(_submit, warmup_images)):
        print(f"    warmup {i + 1}: latency={r['latency_s']:.2f}s")

  print(f"  Measuring ({len(images)} requests, concurrency={concurrent_requests})...")
  latencies: list[float] = []
  t_wall_start = time.perf_counter()

  with ThreadPoolExecutor(max_workers=concurrent_requests) as pool:
    futs = [pool.submit(_submit, img) for img in images]
    for fut in as_completed(futs):
      try:
        r = fut.result()
        latencies.append(r["latency_s"])
      except Exception as e:
        print(f"    request failed: {e}")
      completed = len(latencies)
      if completed % 20 == 0 or completed == len(images):
        elapsed = time.perf_counter() - t_wall_start
        mean_lat = statistics.mean(latencies) if latencies else 0.0
        print(f"    {completed}/{len(images)} done  qps={completed / elapsed:.2f}  mean_latency={mean_lat:.2f}s")

  wall_time = time.perf_counter() - t_wall_start

  def _p(vals: list[float], pct: float) -> float:
    return sorted(vals)[int(len(vals) * pct)]

  return {
    "label": label,
    "n": len(latencies),
    "lat_mean": statistics.mean(latencies),
    "lat_p50": statistics.median(latencies),
    "lat_p95": _p(latencies, 0.95),
    "qps": len(latencies) / wall_time,
    "wall_time_s": wall_time,
  }


def _print_table(stats_list: list[dict[str, Any]]) -> None:
  headers = ["Metric", "Unit"] + [s["label"] for s in stats_list]
  if len(stats_list) == 2:
    headers.append("SGLang faster?")

  def row(metric: str, unit: str, key: str, fmt: str = ".3f", lower_is_better: bool = True) -> list[str]:
    vals = [s[key] for s in stats_list]
    cells = [metric, unit] + [f"{v:{fmt}}" for v in vals]
    if len(stats_list) == 2:
      if lower_is_better:
        winner = "yes" if vals[0] <= vals[1] else "no"
        delta = (vals[1] - vals[0]) / vals[1] * 100 if vals[1] else 0
      else:
        winner = "yes" if vals[0] >= vals[1] else "no"
        delta = (vals[0] - vals[1]) / vals[1] * 100 if vals[1] else 0
      cells.append(f"{winner}  ({delta:+.1f}%)")
    return cells

  rows = [
    row("Latency mean", "s", "lat_mean"),
    row("Latency p50", "s", "lat_p50"),
    row("Latency p95", "s", "lat_p95"),
    row("Throughput (QPS)", "req/s", "qps", ".2f", lower_is_better=False),
    row("Wall time", "s", "wall_time_s", ".1f"),
  ]

  widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
  sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

  def fmt_row(cells: list[str]) -> str:
    return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

  print(sep)
  print(fmt_row(headers))
  print(sep)
  for r in rows:
    print(fmt_row(r))
  print(sep)


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--sglang-url", default="http://localhost:30000")
  parser.add_argument("--vllm-url", default="http://localhost:8000")
  parser.add_argument("--sglang-model", default="Qwen/Qwen3-VL-8B-Instruct")
  parser.add_argument("--vllm-model", default="Qwen/Qwen3-VL-8B-Instruct")
  parser.add_argument(
    "--image-dir",
    default="./images",
    metavar="DIR",
    help="Directory containing JPEG/PNG images to use as benchmark inputs (default: ./images).",
  )
  parser.add_argument("--max-examples", type=int, default=-1, help="Max images to load (-1 for all).")
  parser.add_argument("--max-tokens", type=int, default=1024)
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--concurrent-requests", type=int, default=32)
  parser.add_argument("--request-timeout", type=int, default=600)
  parser.add_argument("--skip-sglang", action="store_true")
  parser.add_argument("--skip-vllm", action="store_true")
  args = parser.parse_args()

  print(f"\n{'=' * 70}")
  print(f"  Image dir   : {args.image_dir}")
  print(f"  Max examples: {args.max_examples}  |  Max tokens: {args.max_tokens}")
  print(f"  Concurrency : {args.concurrent_requests}  |  Warmup: {args.warmup}")
  print(f"{'=' * 70}")

  images = _load_images(args.image_dir, args.max_examples)
  print(f"\nLoaded {len(images)} images from {args.image_dir!r}.")

  all_stats = []

  if not args.skip_sglang:
    stats = _benchmark_server(
      "SGLang", args.sglang_url, args.sglang_model, images,
      DEFAULT_INSTRUCTION, args.max_tokens, args.concurrent_requests,
      args.request_timeout, args.warmup,
    )
    if stats:
      all_stats.append(stats)

  if not args.skip_vllm:
    stats = _benchmark_server(
      "vLLM", args.vllm_url, args.vllm_model, images,
      DEFAULT_INSTRUCTION, args.max_tokens, args.concurrent_requests,
      args.request_timeout, args.warmup,
    )
    if stats:
      all_stats.append(stats)

  if not all_stats:
    print("\nNo servers reachable — skipping.")
    sys.exit(1)

  if args.skip_sglang and args.skip_vllm:
    print("\nBoth servers skipped — nothing to compare.")
    sys.exit(1)

  print(f"\n\n{'=' * 70}")
  print("  Results")
  print(f"{'=' * 70}\n")
  _print_table(all_stats)


if __name__ == "__main__":
  main()
