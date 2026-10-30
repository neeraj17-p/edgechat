"""
eval/run_local.py
─────────────────
Phase 7.2: Local Quantized RAG Pipeline Evaluation Runner.

Executes the EdgeChat local pipeline over all benchmark questions in eval/questions.json:
  - Measures retrieval Hit@1, Hit@3, Hit@5.
  - Measures retrieval latency, generation latency, total latency, and tokens/s.
  - Monitors hardware footprint (RAM and GPU VRAM).
  - Evaluates correct refusal on unanswerable questions.
  - Exports detailed results to eval/results/local_eval_results.csv and .json.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.llm import get_system_memory
from src.prompts import NOT_FOUND_MESSAGE
from src.rag import answer_question, prepare_rag_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval.local")


def run_local_evaluation(
    questions_file: Path | str = "eval/questions.json",
    output_dir: Path | str = "eval/results",
) -> List[Dict[str, Any]]:
    """Run local quantized RAG evaluation on all questions."""
    q_path = Path(questions_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not q_path.is_file():
        raise FileNotFoundError(f"Questions file not found: {q_path}")

    with open(q_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    logger.info("Loaded %d questions from %s", len(questions), q_path)

    results: List[Dict[str, Any]] = []
    total_q = len(questions)

    for idx, item in enumerate(questions, 1):
        q_id = item["q_id"]
        category = item.get("category", "General")
        q_text = item["question"]
        gold_ans = item["gold_answer"]
        gold_doc = item.get("gold_document")
        gold_page = item.get("gold_page")
        is_answerable = item.get("answerable_in_docs", True)

        logger.info(f"[{idx}/{total_q}] Testing {q_id} ({category}): {q_text[:65]}...")

        t0 = time.perf_counter()
        rag_resp = answer_question(q_text)
        wall_time = round(time.perf_counter() - t0, 3)

        mem = get_system_memory()

        # Check retrieval matches
        retrieved_docs = [s.doc_name for s in rag_resp.sources]
        hit_1 = False
        hit_3 = False
        hit_5 = False

        if gold_doc:
            clean_gold = gold_doc.lower().strip()
            # Match against retrieved doc names
            for rank, s in enumerate(rag_resp.sources, 1):
                clean_retrieved = s.doc_name.lower().strip()
                if clean_gold in clean_retrieved or clean_retrieved in clean_gold:
                    if rank == 1:
                        hit_1 = True
                    if rank <= 3:
                        hit_3 = True
                    if rank <= 5:
                        hit_5 = True
                    break

        pred_ans = rag_resp.answer.strip()
        is_refusal = (
            rag_resp.is_refusal
            or pred_ans == NOT_FOUND_MESSAGE
            or NOT_FOUND_MESSAGE.lower() in pred_ans.lower()
        )

        record: Dict[str, Any] = {
            "q_id": q_id,
            "category": category,
            "question": q_text,
            "gold_answer": gold_ans,
            "gold_document": gold_doc or "None",
            "gold_page": gold_page or 0,
            "answerable_in_docs": is_answerable,
            "predicted_answer": pred_ans,
            "is_refusal": is_refusal,
            "retrieved_count": len(rag_resp.sources),
            "top_doc": rag_resp.sources[0].doc_name if rag_resp.sources else "None",
            "top_page": rag_resp.sources[0].page_number if rag_resp.sources else 0,
            "top_score": round(rag_resp.sources[0].score, 4) if rag_resp.sources else 0.0,
            "hit_at_1": hit_1,
            "hit_at_3": hit_3,
            "retrieval_time_s": rag_resp.retrieval_time,
            "generation_time_s": rag_resp.generation_time,
            "total_time_s": rag_resp.total_time,
            "tokens_per_second": rag_resp.tokens_per_second,
            "ram_used_gb": mem["ram_used_gb"],
            "vram_used_mb": mem["vram_used_mb"] or 0.0,
            "model": config.llm.get("model", "llama3.2:3b"),
        }
        results.append(record)

    # Save to JSON
    json_path = out_dir / "local_eval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved local evaluation JSON to %s", json_path)

    # Save to CSV
    csv_path = out_dir / "local_eval_results.csv"
    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    logger.info("Saved local evaluation CSV to %s", csv_path)

    return results


if __name__ == "__main__":
    run_local_evaluation()
