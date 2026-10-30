"""
app.py
──────
Phase 6: Gemini-style Streamlit Interface for EdgeChat.

Design updates:
  - High-contrast, theme-aware typography using Streamlit CSS variables (adapts to light & dark themes).
  - Smooth scrolling behavior across app viewport and chat container.
  - "✦ EdgeChat" in the corner and sidebar.
  - Updated sidebar title & subtitle: "Local RAG Assistant · COEP Regulations, Syllabi & Policies · 100% Offline".
  - One-click prompt cards without page flicker or re-renders.
  - Real-time token-by-token streaming via st.write_stream.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import streamlit as st

from src.config import config
from src.indexer import index_exists, load_index, run_rebuild
from src.llm import generate_stream, get_model_info, get_system_memory
from src.prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT
from src.rag import SourceInfo, prepare_rag_context

# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EdgeChat",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Gemini-inspired Theme-Aware Styling ───────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global typography - remove forced scroll-behavior so native 60/120fps hardware scrolling works without jitter */
    html, body, [data-testid="stAppViewContainer"], section.main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Corner Header Branding */
    .brand-corner {
        margin-top: 0.2rem;
        margin-bottom: 1.25rem;
    }
    .gemini-title {
        font-size: 2.15rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #f8fafc !important; /* Crisp high-contrast white */
        line-height: 1.2;
        margin: 0;
        padding: 0;
    }

    /* Sidebar Tagline / Subtitle */
    .sidebar-tagline {
        font-size: 0.84rem;
        font-weight: 500;
        color: #cbd5e1 !important; /* Crisp light silver, clearly visible */
        opacity: 0.95;
        line-height: 1.45;
        margin-top: -0.4rem;
        margin-bottom: 1.2rem;
    }

    /* Welcome / Empty State Cards */
    .welcome-container {
        padding: 2.4rem 0.5rem 1.4rem 0.5rem;
        text-align: center;
        max-width: 760px;
        margin: 0 auto;
    }
    .welcome-heading {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.4px;
        color: #f8fafc !important; /* Crisp high-contrast white */
        margin-bottom: 0.55rem;
    }
    .welcome-sub {
        font-size: 1.05rem;
        font-weight: 450;
        color: #cbd5e1 !important; /* Crisp light silver, clearly visible */
        opacity: 0.95;
        margin-bottom: 1.9rem;
        line-height: 1.5;
    }

    /* Prompt Cards (Buttons) */
    div[data-testid="stColumn"] button {
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.85rem 1.1rem !important;
        font-size: 0.94rem !important;
        font-weight: 500 !important;
        border-radius: 10px !important;
        min-height: 4.2rem !important;
        transition: all 0.15s ease-in-out !important;
    }

    /* Sources container */
    .source-item {
        background-color: var(--secondary-background-color, rgba(255, 255, 255, 0.05));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.4rem;
        font-size: 0.88rem;
    }
    .source-meta {
        font-weight: 600;
        color: #60a5fa; /* Crisp light blue for dark mode readability */
        margin-bottom: 0.25rem;
    }
    .source-snippet {
        color: #e2e8f0 !important;
        font-style: italic;
    }

    /* Performance latency badge */
    .meta-footer {
        font-size: 0.78rem;
        color: #94a3b8 !important;
        margin-top: 0.4rem;
    }

    /* Optimal content width and breathing space */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5.5rem;
        max-width: 900px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Initialization ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ── Sidebar: Telemetry & Controls ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ✦ EdgeChat")
    st.markdown(
        "<div class='sidebar-tagline'>"
        "Local RAG Assistant · COEP Regulations, Syllabi & Policies · 100% Offline"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Hardware Telemetry**")

    mem = get_system_memory()
    ram_ratio = mem["ram_used_gb"] / mem["ram_total_gb"]
    st.progress(min(1.0, ram_ratio), text=f"RAM: {mem['ram_used_gb']} / {mem['ram_total_gb']} GB ({mem['ram_percent']}%)")

    if mem["vram_used_mb"] is not None and mem["vram_total_mb"] is not None:
        vram_ratio = mem["vram_used_mb"] / mem["vram_total_mb"]
        st.progress(min(1.0, vram_ratio), text=f"GPU VRAM: {mem['vram_used_mb']:.0f} / {mem['vram_total_mb']:.0f} MB")
    else:
        st.caption("GPU: Running on CPU / integrated graphics")

    st.markdown("---")
    st.markdown("**Index & Document Status**")

    if index_exists():
        try:
            faiss_idx, chunks_list = load_index()
            st.write(f"Vectors: `{faiss_idx.ntotal}` · Chunks: `{len(chunks_list)}`")
        except Exception as e:
            st.error(f"Index error: {e}")
    else:
        st.warning("No index found. Click Rebuild Index.")

    model_info = get_model_info()
    st.caption(f"Model: `{model_info['model']}` ({model_info['quantization_level']})")

    # Settings inside a clean expander
    with st.expander("Settings", expanded=False):
        top_k = st.slider(
            "Top-k chunks",
            min_value=1,
            max_value=8,
            value=config.retrieval.get("top_k", 4),
            help="Number of passages retrieved from FAISS.",
        )
        threshold = st.slider(
            "Similarity threshold",
            min_value=0.20,
            max_value=0.85,
            value=float(config.retrieval.get("similarity_threshold", 0.60)),
            step=0.05,
            help="Passages with lower cosine similarity are dropped.",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=0.7,
            value=float(config.llm.get("temperature", 0.1)),
            step=0.05,
            help="Higher = more creative, Lower = deterministic.",
        )

    col_reb, col_clr = st.columns(2)
    with col_reb:
        if st.button("Rebuild Index", use_container_width=True):
            with st.spinner("Rebuilding FAISS index..."):
                try:
                    stats = run_rebuild()
                    st.success(f"Indexed {stats['chunks']} chunks!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")
    with col_clr:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


# ── Main Header Corner Branding ───────────────────────────────────────────────

st.markdown(
    """
    <div class="brand-corner">
        <div class="gemini-title">✦ EdgeChat</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Empty State: Gemini-Style Prompt Cards ────────────────────────────────────

active_query: Optional[str] = None

if len(st.session_state["messages"]) == 0:
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-heading">Where should we start?</div>
            <div class="welcome-sub">Ask anything about academic regulations, syllabus, exam rules, or hostel guidelines.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card_col1, card_col2 = st.columns(2)

    sample_1 = "What is the minimum attendance required for students to be permitted to attend the Mid Semester Examination?"
    sample_2 = "Who is the Vice-Chancellor of COEP Tech University and what is his email address?"
    sample_3 = "Are wrist watches allowed during examinations according to the Code of Conduct for students?"
    sample_4 = "What act prohibits ragging according to the COEP hostel rules and regulations?"

    with card_col1:
        if st.button("✦ Minimum attendance required for Mid Sem Exam", use_container_width=True):
            active_query = sample_1
        if st.button("✦ Exam rules regarding wrist watches", use_container_width=True):
            active_query = sample_3

    with card_col2:
        if st.button("✦ Vice-Chancellor name & contact details", use_container_width=True):
            active_query = sample_2
        if st.button("✦ Hostel ragging prohibition regulations", use_container_width=True):
            active_query = sample_4


# ── Conversation History Display ──────────────────────────────────────────────

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display sources if present
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])} citations)", expanded=False):
                for s_idx, s in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"""
                        <div class="source-item">
                            <div class="source-meta">[{s_idx}] {s['doc_name']} — Page {s['page_number']} (Score: {s['score']:.4f})</div>
                            <div class="source-snippet">"{s['snippet']}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Latency metadata line
        if msg.get("metrics"):
            m = msg["metrics"]
            spd = f"{m['tokens_per_second']:.1f} tok/s · " if m.get("tokens_per_second") else ""
            st.markdown(
                f'<div class="meta-footer">{spd}retrieval {m["retrieval_time"]:.2f}s · generation {m["generation_time"]:.2f}s · total {m["total_time"]:.2f}s</div>',
                unsafe_allow_html=True,
            )


# ── Input Handling & Streaming ────────────────────────────────────────────────

user_input = st.chat_input("Ask a question about college rules, syllabus, or policies...")

if user_input and user_input.strip():
    active_query = user_input.strip()

if active_query:
    # 1. Render and record user message
    st.session_state["messages"].append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # 2. Process and stream assistant response
    with st.chat_message("assistant"):
        t_start = time.perf_counter()

        # Step A: Retrieve context
        with st.spinner("Searching documents..."):
            sources, prompt, retrieval_time = prepare_rag_context(
                question=active_query,
                k=top_k,
                threshold=threshold,
            )

        # Step B: Threshold refusal check
        if prompt is None:
            total_time = round(time.perf_counter() - t_start, 3)
            st.markdown(NOT_FOUND_MESSAGE)
            st.caption("The query similarity score was below the safety threshold, so generation was skipped to prevent hallucinations.")

            st.session_state["messages"].append({
                "role": "assistant",
                "content": NOT_FOUND_MESSAGE,
                "sources": [],
                "metrics": {
                    "retrieval_time": retrieval_time,
                    "generation_time": 0.0,
                    "total_time": total_time,
                    "tokens_per_second": 0.0,
                },
                "is_refusal": True,
            })
        else:
            # Step C: Stream generation token-by-token
            max_tokens = config.generation.get("max_answer_tokens", 512)
            t_gen_start = time.perf_counter()

            try:
                stream_gen = generate_stream(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # st.write_stream renders tokens live and returns full text
                full_answer = st.write_stream(stream_gen)
                gen_time = round(time.perf_counter() - t_gen_start, 3)
                total_time = round(time.perf_counter() - t_start, 3)

                # Estimate tokens and speed
                token_count = len(full_answer.split()) * 4 // 3
                tokens_per_sec = round(token_count / gen_time, 1) if gen_time > 0 else 0.0

                # Render sources drawer under the answer
                sources_dicts = [
                    {
                        "doc_name": s.doc_name,
                        "page_number": s.page_number,
                        "score": s.score,
                        "snippet": s.snippet,
                    }
                    for s in sources
                ]

                with st.expander(f"Sources ({len(sources)} citations)", expanded=False):
                    for s_idx, s in enumerate(sources, 1):
                        st.markdown(
                            f"""
                            <div class="source-item">
                                <div class="source-meta">[{s_idx}] {s.doc_name} — Page {s.page_number} (Score: {s.score:.4f})</div>
                                <div class="source-snippet">"{s.snippet}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # Render subtle latency line
                st.markdown(
                    f'<div class="meta-footer">{tokens_per_sec} tok/s · retrieval {retrieval_time:.2f}s · generation {gen_time:.2f}s · total {total_time:.2f}s</div>',
                    unsafe_allow_html=True,
                )

                # Record in session state
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": full_answer,
                    "sources": sources_dicts,
                    "metrics": {
                        "retrieval_time": retrieval_time,
                        "generation_time": gen_time,
                        "total_time": total_time,
                        "tokens_per_second": tokens_per_sec,
                    },
                    "is_refusal": False,
                })
            except Exception as e:
                st.error(f"Local LLM service error: {e}")
                st.info("Ollama is currently completing its update or model reload. Please retry in a moment.")
                # Pop the user's unanswered question so the UI state remains clean
                if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
                    st.session_state["messages"].pop()
