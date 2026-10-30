"""
tests/test_llm.py
─────────────────
Tests for src/llm.py (Phase 4).

Run from the project root:
    .\\venv\\Scripts\\python.exe -m pytest tests/test_llm.py -v
"""

from __future__ import annotations

import pytest
from src.llm import LLMResponse, generate, get_model_info, get_system_memory


class TestLLMBasics:

    def test_empty_prompt_raises_value_error(self):
        """Calling generate() with an empty or whitespace string must raise ValueError."""
        with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
            generate("")

        with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
            generate("   ")

    def test_get_model_info_returns_metadata(self):
        """get_model_info() must return model name, size, and quantization."""
        info = get_model_info()
        assert isinstance(info, dict)
        assert "llama3.2" in info["model"]
        assert info["size_gb"] > 0
        assert info["format"] == "gguf"

    def test_get_system_memory_returns_ram_stats(self):
        """get_system_memory() must return RAM usage and optional VRAM stats."""
        mem = get_system_memory()
        assert "ram_used_gb" in mem
        assert "ram_total_gb" in mem
        assert mem["ram_used_gb"] > 0
        assert mem["ram_total_gb"] > 0

    def test_generate_simple_prompt(self):
        """generate() should produce a valid LLMResponse with speed metrics."""
        resp = generate(
            prompt="Reply with the single word: OK",
            temperature=0.0,
            max_tokens=10,
        )
        assert isinstance(resp, LLMResponse)
        assert len(resp.text.strip()) > 0
        assert resp.completion_tokens > 0
        assert resp.tokens_per_second > 0.0
        assert resp.total_time > 0.0
        assert "llama3.2" in resp.model
