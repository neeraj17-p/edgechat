"""
eval/eval_retrieval.py
───────────────────────
Phase 3: Retrieval Evaluation (WITHOUT an LLM)

Measures hit@1, hit@3, hit@5, and MRR (Mean Reciprocal Rank) across
10 gold test questions derived directly from the real COEP college documents.

Also evaluates rejection of out-of-domain / unanswerable questions
against the similarity threshold.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.retriever import retrieve, retrieve_for_eval


logging.basicConfig(level=logging.WARNING)

# ── 10 Gold Benchmark Questions from real COEP PDFs ──────────────────────────
# All documents and page numbers are verified against data/index/chunks.json
GOLD_QUESTIONS = [
    {
        "q_id": "Q01",
        "question": "What is the minimum attendance required for students to be permitted to attend the Mid Semester Examination according to UG academic rules?",
        "gold_doc": "UG-Programmes-Academic-Rules.pdf",
        "gold_page": 21,
        "gold_answer": "At least 75% of all conducted classes.",
        "category": "Academic Rules",
    },
    {
        "q_id": "Q02",
        "question": "Who is the Vice-Chancellor of COEP Tech University and what is his email address?",
        "gold_doc": "Vice Chancellor \u2013 COEP Technological University (COEP Tech).pdf",
        "gold_page": 1,
        "gold_answer": "Prof. Sunil Bhirud, Email: vc@coeptech.ac.in",
        "category": "Administration",
    },
    {
        "q_id": "Q03",
        "question": "Who is the Honorable Chancellor of COEP Tech University?",
        "gold_doc": "Chancellor \u2013 COEP Technological University (COEP Tech).pdf",
        "gold_page": 1,
        "gold_answer": "Shri Jishnu Dev Varma, Governor of Maharashtra",
        "category": "Administration",
    },
    {
        "q_id": "Q04",
        "question": "Are wrist watches allowed during examinations according to the Code of Conduct for students?",
        "gold_doc": "COEP_Code-of-Conduct-during-Examination-for-Students-2024.pdf",
        "gold_page": 1,
        "gold_answer": "Strictly, wrist watches of any type are not allowed.",
        "category": "Exam Rules",
    },
    {
        "q_id": "Q05",
        "question": "What act prohibits ragging according to the COEP hostel rules and regulations?",
        "gold_doc": "Hostel-Rules-and-Regulation._COEP-Tech.-2025.pdf",
        "gold_page": 1,
        "gold_answer": "Maharashtra Prohibition of Ragging Act, 1999 (Maharashtra Act No. XXXIII of 1999)",
        "category": "Hostel Rules",
    },
    {
        "q_id": "Q06",
        "question": "What is the vision of COEP Technological University?",
        "gold_doc": "Vision and Mission \u2013 COEP Technological University (COEP Tech).pdf",
        "gold_page": 1,
        "gold_answer": "To be a Value-based Globally Recognized Institution ensuring academic excellence and fostering Research, Innovation, and Entrepreneurial Attitude.",
        "category": "General",
    },
    {
        "q_id": "Q07",
        "question": "Who served as the Vice Chancellor of COEP from 21.06.2022 to 17.04.2023?",
        "gold_doc": "Former Vice Chancellors \u2013 COEP Technological University (COEP Tech).pdf",
        "gold_page": 1,
        "gold_answer": "Prof. Mukul Sutaone",
        "category": "History",
    },
    {
        "q_id": "Q08",
        "question": "Who is the Chief Rector (Incharge) of COEP Tech University and what is his email address?",
        "gold_doc": "Administrative Staff Contact Information \u2013 COEP Technological University (COEP Tech).pdf",
        "gold_page": 1,
        "gold_answer": "Dr. Nitin M. Mohite, chiefrector@coeptech.ac.in",
        "category": "Administration",
    },
    {
        "q_id": "Q09",
        "question": "When is the Semester Credit Registration scheduled for Odd Semester Term-I in AY 2026-27 for UG and PG?",
        "gold_doc": "Academic Calendar for UG SY, TY,  Final Year B-TECH and SY M-TECH 2026-27.pdf",
        "gold_page": 1,
        "gold_answer": "2nd July 2026 (Thursday) to 3rd July 2026 (Friday)",
        "category": "Academic Calendar",
    },
    {
        "q_id": "Q10",
        "question": "For education loans taken after which date is the PM-Vidyalaxmi scheme guidelines applicable?",
        "gold_doc": "PM_Vidyalaxmi_Scheme_Guidelines.pdf",
        "gold_page": 1,
        "gold_answer": "Applicable for Education Loans taken after 6th November 2024",
        "category": "Scholarship/Scheme",
    },
]

# 2 Negative / Out-of-Domain questions to test threshold rejection
UNANSWERABLE_QUESTIONS = [
    {
        "q_id": "U01",
        "question": "What is the launch date of the Apollo 11 moon mission and who was the pilot?",
        "expected_result": "empty",
    },
    {
        "q_id": "U02",
        "question": "How do you calculate the Schwarzschild radius of a rotating Kerr black hole?",
        "expected_result": "empty",
    },
]


def run_evaluation():
    print("\n" + "=" * 78)
    print("  EDGECHAT PHASE 3 RETRIEVAL BENCHMARK")
    print("  Evaluating Hit@1, Hit@3, Hit@5, MRR on 10 Verified Real Documents")
    print("=" * 78)

    results = []
    hit_1_count = 0
    hit_3_count = 0
    hit_5_count = 0
    reciprocal_ranks = []

    for item in GOLD_QUESTIONS:
        qid = item["q_id"]
        q = item["question"]
        gold_doc = item["gold_doc"]
        gold_page = item["gold_page"]

        retrieved = retrieve_for_eval(q, k=5)

        hit_1 = False
        hit_3 = False
        hit_5 = False
        first_gold_rank = None

        for idx, r in enumerate(retrieved):
            rank = idx + 1
            # Check match by doc and page (or doc if page numbers align)
            doc_match = (r.doc_name == gold_doc)
            page_match = (r.page_number == gold_page)

            if doc_match and page_match:
                if first_gold_rank is None:
                    first_gold_rank = rank
                if rank == 1:
                    hit_1 = True
                if rank <= 3:
                    hit_3 = True
                if rank <= 5:
                    hit_5 = True

        if hit_1:
            hit_1_count += 1
        if hit_3:
            hit_3_count += 1
        if hit_5:
            hit_5_count += 1

        rr = (1.0 / first_gold_rank) if first_gold_rank else 0.0
        reciprocal_ranks.append(rr)

        top1_doc = retrieved[0].doc_name if retrieved else "None"
        top1_page = retrieved[0].page_number if retrieved else 0
        top1_score = retrieved[0].score if retrieved else 0.0

        results.append({
            "q_id": qid,
            "category": item["category"],
            "question": q,
            "gold_doc": gold_doc,
            "gold_page": gold_page,
            "top1_doc": top1_doc,
            "top1_page": top1_page,
            "top1_score": round(top1_score, 4),
            "gold_rank": first_gold_rank if first_gold_rank else "Not in top-5",
            "hit@1": "YES" if hit_1 else "NO",
            "hit@3": "YES" if hit_3 else "NO",
            "hit@5": "YES" if hit_5 else "NO",
        })

    # Summary metrics
    n = len(GOLD_QUESTIONS)
    hit_1_rate = (hit_1_count / n) * 100
    hit_3_rate = (hit_3_count / n) * 100
    hit_5_rate = (hit_5_count / n) * 100
    mrr = sum(reciprocal_ranks) / n

    df = pd.DataFrame(results)
    
    print("\n--- PER-QUESTION RETRIEVAL BREAKDOWN ---")
    for _, row in df.iterrows():
        print(f"\n[{row['q_id']}] {row['question']}")
        print(f"  Target: {row['gold_doc']} (Page {row['gold_page']})")
        print(f"  Top-1:  {row['top1_doc']} (Page {row['top1_page']}) | Score: {row['top1_score']:.4f}")
        print(f"  First Gold Chunk Rank: {row['gold_rank']} | Hit@1: {row['hit@1']} | Hit@3: {row['hit@3']} | Hit@5: {row['hit@5']}")

    print("\n" + "=" * 78)
    print("  RETRIEVAL METRICS SUMMARY")
    print("=" * 78)
    print(f"  Total Questions: {n}")
    print(f"  Hit@1:           {hit_1_count}/{n}  ({hit_1_rate:.1f}%)")
    print(f"  Hit@3:           {hit_3_count}/{n}  ({hit_3_rate:.1f}%)")
    print(f"  Hit@5:           {hit_5_count}/{n}  ({hit_5_rate:.1f}%)")
    print(f"  MRR (Mean RR):   {mrr:.4f}")
    print("=" * 78)

    # Negative / Threshold testing
    print("\n--- OUT-OF-DOMAIN / UNANSWERABLE THRESHOLD TEST ---")
    threshold_passed = True
    for u in UNANSWERABLE_QUESTIONS:
        res = retrieve(u["question"])  # uses default threshold (0.30)
        is_empty = (len(res) == 0)
        status = "PASSED (Rejected)" if is_empty else f"FAILED (Retrieved {len(res)} chunks)"
        print(f"  [{u['q_id']}] {u['question'][:65]}... -> {status}")
        if not is_empty:
            threshold_passed = False

    # Save to eval/results
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "retrieval_benchmark.csv"
    json_path = results_dir / "retrieval_benchmark.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "total_questions": n,
                "hit@1_rate": hit_1_rate,
                "hit@3_rate": hit_3_rate,
                "hit@5_rate": hit_5_rate,
                "mrr": round(mrr, 4),
                "threshold_passed": threshold_passed,
            },
            "per_question": results,
        }, f, indent=2)

    print(f"\n[OK] Results saved to:\n  - {csv_path}\n  - {json_path}\n")
    return hit_1_rate, hit_3_rate, hit_5_rate, mrr


if __name__ == "__main__":
    run_evaluation()
