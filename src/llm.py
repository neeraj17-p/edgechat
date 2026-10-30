"""
src/llm.py
──────────
Phase 4: Local Quantized LLM Runtime (Ollama).

Connects to a local Ollama instance running a quantized GGUF model
(default: llama3.2:3b Q4_K_M).

Features:
  - Measures latency, tokens per second, load duration, prompt eval time.
  - Monitors system RAM and GPU VRAM usage.
  - Catches connection errors with helpful troubleshooting guidance.
  - Reads defaults (temperature, num_ctx, seed) from config.yaml.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import psutil
try:
    import ollama
except ImportError:
    ollama = None

from src.config import config

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """
    Structured response from the local LLM.

    Attributes:
        text:               The generated text response.
        model:              Model identifier used (e.g., "llama3.2:3b").
        prompt_tokens:      Number of tokens in the prompt.
        completion_tokens:  Number of generated tokens.
        total_time:         Total round-trip time in seconds.
        load_time:          Time spent loading the model into memory in seconds.
        prompt_eval_time:   Time spent processing the prompt in seconds.
        eval_time:          Time spent generating completion tokens in seconds.
        tokens_per_second:  Generation speed (completion_tokens / eval_time).
    """
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_time: float
    load_time: float
    prompt_eval_time: float
    eval_time: float
    tokens_per_second: float


# ── Memory & Diagnostic Helpers ───────────────────────────────────────────────

def get_system_memory() -> dict:
    """
    Measure current system RAM and GPU VRAM usage.

    Returns:
        Dict with keys:
          - ram_used_gb, ram_total_gb, ram_percent
          - vram_used_mb, vram_total_mb (or None if no NVIDIA GPU)
    """
    ram = psutil.virtual_memory()
    info = {
        "ram_used_gb": round(ram.used / (1024 ** 3), 2),
        "ram_total_gb": round(ram.total / (1024 ** 3), 2),
        "ram_percent": ram.percent,
        "vram_used_mb": None,
        "vram_total_mb": None,
    }

    # Query NVIDIA GPU memory if available via nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
        if out:
            used, total = out.split("\n")[0].split(",")
            info["vram_used_mb"] = float(used.strip())
            info["vram_total_mb"] = float(total.strip())
    except Exception:
        pass  # GPU query is optional; CPU systems will report None

    return info


def get_model_info(model_name: Optional[str] = None) -> dict:
    """
    Retrieve metadata for the installed Ollama model.

    Returns:
        Dict with size, parameter size, quantization level, and format.
    """
    if ollama is None:
        raise RuntimeError("The 'ollama' Python package is not installed.")

    model_name = model_name or config.llm["model"]
    try:
        models_resp = ollama.list()
        # Find matching model (handles 'llama3.2:3b' or 'llama3.2:3b:latest')
        target_base = model_name.split(":")[0]
        for m in getattr(models_resp, "models", []):
            m_name = getattr(m, "model", "")
            if m_name == model_name or m_name.startswith(f"{target_base}:"):
                size_bytes = getattr(m, "size", 0)
                details = getattr(m, "details", None)
                return {
                    "model": m_name,
                    "size_bytes": size_bytes,
                    "size_gb": round(size_bytes / (1024 ** 3), 2),
                    "parameter_size": getattr(details, "parameter_size", "unknown") if details else "unknown",
                    "quantization_level": getattr(details, "quantization_level", "unknown") if details else "unknown",
                    "format": getattr(details, "format", "gguf") if details else "gguf",
                }
    except Exception as e:
        logger.warning("Could not query model info from Ollama: %s", e)

    return {
        "model": model_name,
        "size_bytes": 0,
        "size_gb": 0.0,
        "parameter_size": "unknown",
        "quantization_level": "unknown",
        "format": "gguf",
    }


# ── Core Generation ───────────────────────────────────────────────────────────

def generate(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    num_ctx: Optional[int] = None,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> LLMResponse:
    """
    Generate a text response from the local Ollama LLM.

    Args:
        prompt:      The user / context prompt string.
        system:      Optional system prompt instruction.
        model:       Model name (defaults to config.yaml value, e.g. "llama3.2:3b").
        temperature: Sampling temperature (defaults to config.yaml value, e.g. 0.1).
        num_ctx:     Context window size (defaults to config.yaml value, e.g. 4096).
        seed:        Random seed for reproducibility (defaults to config.yaml value).
        max_tokens:  Maximum tokens to generate (optional).

    Returns:
        LLMResponse with text and timing / speed metrics.

    Raises:
        ValueError:   If prompt is empty.
        RuntimeError: If Ollama service is unreachable or model fails.
    """
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt must be a non-empty string.")

    if ollama is None:
        raise RuntimeError("The 'ollama' Python package is not installed.")

    # Defaults from config
    model = model or config.llm["model"]
    temperature = temperature if temperature is not None else config.llm.get("temperature", 0.1)
    num_ctx = num_ctx if num_ctx is not None else config.llm.get("num_ctx", 4096)
    seed = seed if seed is not None else config.llm.get("seed", 42)

    options = {
        "temperature": float(temperature),
        "num_ctx": int(num_ctx),
        "seed": int(seed),
    }
    if max_tokens is not None:
        options["num_predict"] = int(max_tokens)

    t_start = time.perf_counter()

    try:
        raw_res = ollama.generate(
            model=model,
            prompt=prompt,
            system=system,
            options=options,
        )
    except Exception as e:
        err_msg = str(e)
        logger.error("Ollama generate failed: %s", err_msg)
        raise RuntimeError(
            f"Failed to communicate with local Ollama service for model {model!r}.\n"
            f"Please ensure Ollama is running (`ollama list` in terminal).\n"
            f"Error details: {err_msg}"
        ) from e

    wall_time = time.perf_counter() - t_start

    # Extract Ollama timing statistics (durations are reported in nanoseconds)
    eval_count = getattr(raw_res, "eval_count", 0) or 0
    prompt_eval_count = getattr(raw_res, "prompt_eval_count", 0) or 0

    load_duration_ns = getattr(raw_res, "load_duration", 0) or 0
    prompt_eval_duration_ns = getattr(raw_res, "prompt_eval_duration", 0) or 0
    eval_duration_ns = getattr(raw_res, "eval_duration", 0) or 0
    total_duration_ns = getattr(raw_res, "total_duration", 0) or 0

    load_time = load_duration_ns / 1e9
    prompt_eval_time = prompt_eval_duration_ns / 1e9
    eval_time = eval_duration_ns / 1e9
    total_time = (total_duration_ns / 1e9) if total_duration_ns > 0 else wall_time

    tokens_per_sec = (eval_count / eval_time) if eval_time > 0 else 0.0

    text = getattr(raw_res, "response", "").strip()

    logger.info(
        "LLM generated %d tokens in %.2fs (%.1f tok/s, prompt_eval=%.2fs, load=%.2fs)",
        eval_count, eval_time, tokens_per_sec, prompt_eval_time, load_time,
    )

    return LLMResponse(
        text=text,
        model=model,
        prompt_tokens=prompt_eval_count,
        completion_tokens=eval_count,
        total_time=round(total_time, 3),
        load_time=round(load_time, 3),
        prompt_eval_time=round(prompt_eval_time, 3),
        eval_time=round(eval_time, 3),
        tokens_per_second=round(tokens_per_sec, 2),
    )


def generate_stream(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    num_ctx: Optional[int] = None,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = None,
):
    """
    Stream tokens from the local Ollama LLM as they are generated.

    Yields:
        str: Token strings for real-time rendering.
    """
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt must be a non-empty string.")

    if ollama is None:
        raise RuntimeError("The 'ollama' Python package is not installed.")

    model = model or config.llm["model"]
    temperature = temperature if temperature is not None else config.llm.get("temperature", 0.1)
    num_ctx = num_ctx if num_ctx is not None else config.llm.get("num_ctx", 4096)
    seed = seed if seed is not None else config.llm.get("seed", 42)

    options = {
        "temperature": float(temperature),
        "num_ctx": int(num_ctx),
        "seed": int(seed),
    }
    if max_tokens is not None:
        options["num_predict"] = int(max_tokens)

    try:
        raw_stream = ollama.generate(
            model=model,
            prompt=prompt,
            system=system,
            options=options,
            stream=True,
        )
        for chunk in raw_stream:
            token = getattr(chunk, "response", "")
            if token:
                yield token
    except Exception as e:
        err_msg = str(e)
        logger.error("Ollama generate_stream failed: %s", err_msg)
        raise RuntimeError(
            f"Failed to stream from local Ollama service for model {model!r}: {err_msg}"
        ) from e


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="EdgeChat Local LLM CLI")
    parser.add_argument("prompt", nargs="?", help="Prompt to send to the local model")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--temp", type=float, default=None, help="Temperature (0.0 to 1.0)")
    parser.add_argument("--system", type=str, default=None, help="System prompt")

    args = parser.parse_args()

    # Query model info
    info = get_model_info(args.model)
    mem = get_system_memory()

    print("\n" + "=" * 65)
    print(f"  EdgeChat Local LLM — Model: {info['model']}")
    print(f"  Params: {info['parameter_size']} | Quant: {info['quantization_level']} | Size: {info['size_gb']} GB")
    vram_str = f" | VRAM: {mem['vram_used_mb']:.0f}/{mem['vram_total_mb']:.0f} MB" if mem['vram_used_mb'] else ""
    print(f"  RAM: {mem['ram_used_gb']}/{mem['ram_total_gb']} GB ({mem['ram_percent']}used){vram_str}")
    print("=" * 65)

    prompt = args.prompt or "Explain what Retrieval-Augmented Generation (RAG) is in two sentences."
    print(f"\nPrompt: {prompt}\n")

    resp = generate(
        prompt=prompt,
        system=args.system,
        model=args.model,
        temperature=args.temp,
    )

    print(f"Response:\n{resp.text}\n")
    print("-" * 65)
    print(f"Stats: {resp.completion_tokens} tokens | {resp.tokens_per_second:.1f} tok/s | eval: {resp.eval_time}s | total: {resp.total_time}s")
    print("=" * 65 + "\n")
