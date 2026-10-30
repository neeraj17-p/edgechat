"""
eval/benchmark_llm.py
──────────────────────
Phase 4: Benchmark Local Quantized LLM (Ollama).

Measures and records:
  - Model file size (GB)
  - Model load time (seconds)
  - Prompt processing time (seconds)
  - Generation speed (tokens per second)
  - Total latency (seconds)
  - System RAM usage (GB, %)
  - GPU VRAM usage (MB, %)
Saves all measurements to eval/results/llm_benchmark.csv and .json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.llm import generate, get_model_info, get_system_memory

logging.basicConfig(level=logging.WARNING)

BENCHMARK_PROMPTS = [
    {
        "type": "Short Q&A",
        "prompt": "What is the capital of France and what is its most famous landmark? Answer in one concise sentence.",
    },
    {
        "type": "Medium Explanation",
        "prompt": (
            "Explain in three concise bullet points why maintaining 75% attendance "
            "is essential for engineering students in college."
        ),
    },
    {
        "type": "Simulated RAG Context",
        "prompt": (
            "You are an academic assistant. Answer the question based ONLY on the context below.\n\n"
            "CONTEXT:\n"
            "COEP Technological University Pune regulations state: Each student shall be required "
            "to attend at least 75% of all conducted classes like lectures, tutorials, laboratories, "
            "studios, and workshops for being permitted to attend the Mid Semester Examination (MSE). "
            "Students failing to achieve 75% attendance will be detained and awarded an 'XX' grade.\n\n"
            "QUESTION:\n"
            "What happens if a student fails to achieve 75% attendance?"
        ),
    },
]


def run_benchmark():
    print("\n" + "=" * 78)
    print("  EDGECHAT PHASE 4: LOCAL QUANTIZED LLM BENCHMARK")
    print("  Measuring Load Time, Generation Speed, RAM & VRAM Usage")
    print("=" * 78)

    model_info = get_model_info()
    print(f"\n  Model:         {model_info['model']}")
    print(f"  Format:        {model_info['format'].upper()} (Quantization: {model_info['quantization_level']})")
    print(f"  Parameters:    {model_info['parameter_size']}")
    print(f"  File Size:     {model_info['size_gb']:.2f} GB ({model_info['size_bytes']:,} bytes)")

    mem_before = get_system_memory()
    vram_str = f"{mem_before['vram_used_mb']:.0f} / {mem_before['vram_total_mb']:.0f} MB" if mem_before['vram_used_mb'] else "N/A (CPU only)"
    print(f"  Initial RAM:   {mem_before['ram_used_gb']:.2f} / {mem_before['ram_total_gb']:.2f} GB ({mem_before['ram_percent']}%)")
    print(f"  Initial VRAM:  {vram_str}")
    print("=" * 78 + "\n")

    records = []

    # Run 3 trials per prompt
    total_runs = len(BENCHMARK_PROMPTS) * 3
    run_idx = 1

    for item in BENCHMARK_PROMPTS:
        p_type = item["type"]
        prompt = item["prompt"]

        print(f"Testing Prompt Category: [{p_type}]")

        for trial in range(1, 4):
            mem_current = get_system_memory()
            resp = generate(prompt, temperature=0.1)

            rec = {
                "run_id": run_idx,
                "prompt_type": p_type,
                "trial": trial,
                "model": resp.model,
                "parameter_size": model_info["parameter_size"],
                "quantization": model_info["quantization_level"],
                "model_size_gb": model_info["size_gb"],
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "tokens_per_sec": resp.tokens_per_second,
                "total_time_s": resp.total_time,
                "eval_time_s": resp.eval_time,
                "prompt_eval_time_s": resp.prompt_eval_time,
                "load_time_s": resp.load_time,
                "ram_used_gb": mem_current["ram_used_gb"],
                "ram_percent": mem_current["ram_percent"],
                "vram_used_mb": mem_current["vram_used_mb"],
                "vram_total_mb": mem_current["vram_total_mb"],
                "response_sample": resp.text.replace("\n", " ")[:120],
            }
            records.append(rec)

            print(
                f"  Trial {trial}/3: {resp.completion_tokens:3d} tokens | "
                f"{resp.tokens_per_second:5.1f} tok/s | "
                f"total: {resp.total_time:5.2f}s | "
                f"RAM: {mem_current['ram_used_gb']:.1f} GB"
                + (f" | VRAM: {mem_current['vram_used_mb']:.0f} MB" if mem_current['vram_used_mb'] else "")
            )
            run_idx += 1

        print()

    df = pd.DataFrame(records)

    # Compute summary aggregates
    avg_speed = df["tokens_per_sec"].mean()
    min_speed = df["tokens_per_sec"].min()
    max_speed = df["tokens_per_sec"].max()
    avg_latency = df["total_time_s"].mean()
    avg_load_time = df["load_time_s"].mean()
    peak_ram = df["ram_used_gb"].max()
    peak_vram = df["vram_used_mb"].max() if mem_before["vram_used_mb"] else None

    print("=" * 78)
    print("  PHASE 4 BENCHMARK SUMMARY")
    print("=" * 78)
    print(f"  Model Name:            {model_info['model']} ({model_info['quantization_level']})")
    print(f"  Model File Size:       {model_info['size_gb']:.2f} GB")
    print(f"  Average Speed:         {avg_speed:.1f} tokens/second (range: {min_speed:.1f} - {max_speed:.1f})")
    print(f"  Average Latency:       {avg_latency:.2f} seconds")
    print(f"  Average Load Time:     {avg_load_time:.3f} seconds (in-memory cached by Ollama)")
    print(f"  Peak RAM Usage:        {peak_ram:.2f} GB")
    if peak_vram:
        print(f"  Peak GPU VRAM Usage:   {peak_vram:.0f} MB")
    print("=" * 78)

    # Save to eval/results
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "llm_benchmark.csv"
    json_path = results_dir / "llm_benchmark.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_info": model_info,
            "summary": {
                "avg_tokens_per_sec": round(avg_speed, 2),
                "min_tokens_per_sec": round(min_speed, 2),
                "max_tokens_per_sec": round(max_speed, 2),
                "avg_latency_s": round(avg_latency, 3),
                "avg_load_time_s": round(avg_load_time, 3),
                "peak_ram_gb": round(peak_ram, 2),
                "peak_vram_mb": round(peak_vram, 2) if peak_vram else None,
            },
            "runs": records,
        }, f, indent=2)

    print(f"\n[OK] Benchmark results recorded in:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}\n")

    return df


if __name__ == "__main__":
    run_benchmark()
