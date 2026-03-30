"""
Meeting Summarizer — Streamlit App
Built on: philschmid/bart-large-cnn-samsum (BART + SAMSum fine-tuned)

Run with:
    pip install streamlit transformers torch
    streamlit run meeting_summarizer.py
"""

import re
import time
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="MeetingMind · AI Summarizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

/* Root reset */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* App background */
.stApp {
    background: #F8FAFF;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E8ECF4;
}
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 13px;
    color: #6B7280;
}

/* ── MAIN HEADER ── */
.app-header {
    background: #FFFFFF;
    border: 1px solid #E8ECF4;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.app-header h1 {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.app-header p {
    font-size: 14px;
    color: #6B7280;
    margin: 0;
}
.brand-tag {
    display: inline-block;
    background: #EEF2FF;
    color: #4F46E5;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
}

/* ── CARDS ── */
.card {
    background: #FFFFFF;
    border: 1px solid #E8ECF4;
    border-radius: 14px;
    padding: 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.card-title {
    font-size: 13px;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── SUMMARY SECTION PILLS ── */
.section-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}
.pill-overview  { background: #F0F9FF; color: #0369A1; }
.pill-problems  { background: #FEF2F2; color: #DC2626; }
.pill-decisions { background: #F0FDF4; color: #16A34A; }
.pill-actions   { background: #FFFBEB; color: #D97706; }
.pill-deadlines { background: #FDF4FF; color: #9333EA; }

/* ── SUMMARY ITEMS ── */
.summary-item {
    background: #F9FAFB;
    border-left: 3px solid #E5E7EB;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13.5px;
    color: #374151;
    line-height: 1.65;
}
.summary-item.overview  { border-left-color: #0EA5E9; }
.summary-item.problems  { border-left-color: #EF4444; }
.summary-item.decisions { border-left-color: #22C55E; }
.summary-item.actions   { border-left-color: #F59E0B; }
.summary-item.deadlines { border-left-color: #A855F7; }

/* ── RAW SUMMARY BOX ── */
.raw-summary {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 18px 20px;
    font-size: 14px;
    line-height: 1.8;
    color: #1F2937;
    font-family: 'Sora', sans-serif;
}

/* ── STAT BADGES ── */
.stat-row {
    display: flex;
    gap: 12px;
    margin: 12px 0 0 0;
    flex-wrap: wrap;
}
.stat-badge {
    background: #F3F4F6;
    border-radius: 8px;
    padding: 8px 16px;
    text-align: center;
    min-width: 90px;
}
.stat-value {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    line-height: 1.1;
}
.stat-label {
    font-size: 10px;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 2px;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── BUTTONS ── */
.stButton > button {
    background: #111827 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #1F2937 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

/* ── TEXT AREA ── */
.stTextArea textarea {
    font-family: 'Sora', sans-serif !important;
    font-size: 13.5px !important;
    border-radius: 10px !important;
    border: 1px solid #D1D5DB !important;
    color: #111827 !important;
    line-height: 1.7 !important;
}
.stTextArea textarea:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* ── SLIDERS & SELECTS ── */
.stSlider > div { padding-top: 4px; }

/* ── DIVIDER ── */
hr { border-color: #E8ECF4 !important; margin: 20px 0 !important; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #6366F1 !important; }

/* ── SUCCESS / INFO MESSAGES ── */
.stSuccess, .stInfo {
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
}

/* ── EMPTY STATE ── */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: #9CA3AF;
}
.empty-state .icon { font-size: 40px; margin-bottom: 12px; }
.empty-state p { font-size: 14px; margin: 0; }
</style>
""", unsafe_allow_html=True)


# ─── CONFIG ─────────────────────────────────────────────────────
MODEL_NAME = "philschmid/bart-large-cnn-samsum"
DEFAULT_MAX_INPUT  = 800
DEFAULT_STRIDE     = 100
DEFAULT_SUMMARY_LEN = 150


# ─── MODEL LOADING (cached) ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


# ─── PROCESSING FUNCTIONS ────────────────────────────────────────
def clean_dialogue(text: str) -> str:
    text = re.sub(r"Speaker [A-Z]:", "", text)
    text = re.sub(r"\b(uh|um|ah|hmm|er)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, tokenizer, max_tokens: int, stride: int):
    tokens = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), max_tokens - stride):
        chunk = tokenizer.decode(tokens[i:i + max_tokens], skip_special_tokens=True)
        chunks.append(chunk)
    return chunks


def summarize_chunk(chunk: str, tokenizer, model, device, max_len: int) -> str:
    inputs = tokenizer(chunk, return_tensors="pt", max_length=900, truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=4,
            max_length=max_len,
            early_stopping=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def run_summarize(text: str, tokenizer, model, device,
                  max_input: int, stride: int, max_summary: int,
                  prompt_prefix: str) -> tuple[str, int, float]:
    start = time.time()
    text = prompt_prefix + clean_dialogue(text)
    chunks = chunk_text(text, tokenizer, max_input, stride)
    summaries = [summarize_chunk(c, tokenizer, model, device, max_summary) for c in chunks]
    final = " ".join(summaries)
    if len(tokenizer(final)["input_ids"]) > max_input:
        final = summarize_chunk(final, tokenizer, model, device, max_summary)
    elapsed = round(time.time() - start, 2)
    return final, len(chunks), elapsed


def classify_summary(summary: str) -> dict:
    sentences = [s.strip() for s in summary.split(". ") if s.strip()]
    result = {"overview": [], "problems": [], "decisions": [], "actions": [], "deadlines": []}

    for s in sentences:
        low = s.lower()
        if any(k in low for k in ["issue", "problem", "slow", "error", "delay", "not working", "performance"]):
            result["problems"].append(s)
        elif any(k in low for k in ["day", "deadline", "scheduled", "due", "within", "by next", "by end"]):
            result["deadlines"].append(s)
        elif any(k in low for k in ["decided", "plan", "approach", "solution", "will", "should", "going to", "agreed"]):
            result["decisions"].append(s)
        elif any(k in low for k in ["handle", "work on", "responsible", "assigned", "implement", "review", "prepare"]):
            result["actions"].append(s)
        else:
            result["overview"].append(s)

    if not result["decisions"]:
        result["decisions"] = result["overview"][:1]
    if not result["actions"]:
        result["actions"] = result["decisions"][:1]

    return result


# ─── SIDEBAR ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    st.markdown("---")

    max_input_tokens = st.slider(
        "Max input tokens per chunk", 400, 1024, DEFAULT_MAX_INPUT, step=50,
        help="Controls how much text each chunk contains before being sent to the model."
    )
    stride_tokens = st.slider(
        "Overlap (stride) tokens", 50, 300, DEFAULT_STRIDE, step=25,
        help="How many tokens overlap between consecutive chunks to preserve context."
    )
    max_summary_len = st.slider(
        "Max summary length (tokens)", 60, 300, DEFAULT_SUMMARY_LEN, step=10,
        help="Maximum number of tokens in the generated summary."
    )

    st.markdown("---")
    st.markdown("### 🎯 Prompt Prefix")
    prompt_prefix = st.text_area(
        "Custom instruction (prepended to input)",
        value=(
            "Summarize the meeting professionally.\n"
            "Do NOT mention speakers.\n"
            "Focus on problems, decisions, actions, and deadlines:\n\n"
        ),
        height=120,
        help="This instruction is prepended to your transcript before summarization."
    )

    st.markdown("---")
    st.markdown("### 📦 Model")
    st.markdown(f"`{MODEL_NAME}`")
    st.markdown("BART large, fine-tuned on SAMSum conversational summarization dataset.")

    st.markdown("---")
    st.markdown(
        "<p style='font-size:11px;color:#9CA3AF;text-align:center;'>"
        "Built with 🤗 Transformers + Streamlit</p>",
        unsafe_allow_html=True
    )


# ─── HEADER ──────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="brand-tag">NLP · BART · SAMSum</div>
  <h1>🧠 MeetingMind</h1>
  <p>Paste your meeting transcript below. The AI will extract key decisions, action items, problems, and deadlines automatically.</p>
</div>
""", unsafe_allow_html=True)


# ─── INPUT + TRIGGER ─────────────────────────────────────────────
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="card-title">📋 Transcript Input</div>', unsafe_allow_html=True)

    transcript = st.text_area(
        label="Meeting transcript",
        placeholder=(
            "Paste your meeting transcript here…\n\n"
            "Example:\n"
            "Speaker A: The server response time is way too slow.\n"
            "Speaker B: Agreed. We should switch to async processing.\n"
            "Speaker A: Let's aim to have a fix by end of week.\n"
        ),
        height=320,
        label_visibility="collapsed",
    )

    word_count = len(transcript.split()) if transcript.strip() else 0
    char_count = len(transcript)

    # Stats row
    if transcript.strip():
        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-badge">
            <div class="stat-value">{word_count}</div>
            <div class="stat-label">Words</div>
          </div>
          <div class="stat-badge">
            <div class="stat-value">{char_count}</div>
            <div class="stat-label">Chars</div>
          </div>
          <div class="stat-badge">
            <div class="stat-value">{len(transcript.splitlines())}</div>
            <div class="stat-label">Lines</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⚡ Summarize Meeting", use_container_width=True)

    # Sample transcript loader
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📄 Load a sample transcript"):
        sample = """Speaker A: The backend API is responding too slowly — averaging 4 seconds per request.
Speaker B: Yes, I noticed that too. The database queries aren't optimized.
Speaker C: We should implement query caching and use an async task queue.
Speaker A: Agreed. That's our main approach going forward.
Speaker B: I'll take ownership of the caching layer. Should be ready by Thursday.
Speaker C: I'll handle the async queue setup. Aiming for end of this week.
Speaker A: Great. We also need to write unit tests for the new components.
Speaker B: I'll include test coverage as part of the caching PR.
Speaker C: Noted. Let's schedule a review session on Friday afternoon to check progress."""
        if st.button("Load Sample →", key="sample_btn"):
            st.session_state["loaded_sample"] = sample
            st.rerun()

    if "loaded_sample" in st.session_state:
        transcript = st.session_state.pop("loaded_sample")


# ─── OUTPUT ──────────────────────────────────────────────────────
with col_output:
    st.markdown('<div class="card-title">📊 Summary Output</div>', unsafe_allow_html=True)

    if run_btn and transcript.strip():
        with st.spinner("Loading model & summarizing…"):
            model, tokenizer, device = load_model()
            summary, n_chunks, elapsed = run_summarize(
                transcript, tokenizer, model, device,
                max_input_tokens, stride_tokens, max_summary_len, prompt_prefix
            )

        st.session_state["summary"]  = summary
        st.session_state["n_chunks"] = n_chunks
        st.session_state["elapsed"]  = elapsed

    elif run_btn and not transcript.strip():
        st.warning("⚠️ Please paste a transcript first.")

    if "summary" in st.session_state:
        summary   = st.session_state["summary"]
        n_chunks  = st.session_state["n_chunks"]
        elapsed   = st.session_state["elapsed"]
        classified = classify_summary(summary)

        # Meta stats
        st.markdown(f"""
        <div class="stat-row" style="margin-bottom:20px;">
          <div class="stat-badge">
            <div class="stat-value">{n_chunks}</div>
            <div class="stat-label">Chunks</div>
          </div>
          <div class="stat-badge">
            <div class="stat-value">{elapsed}s</div>
            <div class="stat-label">Time</div>
          </div>
          <div class="stat-badge">
            <div class="stat-value">{len(summary.split())}</div>
            <div class="stat-label">Words</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Structured output
        sections = [
            ("overview",   "🔵 Overview",   "pill-overview",  "overview"),
            ("problems",   "🔴 Problems",   "pill-problems",  "problems"),
            ("decisions",  "🟢 Decisions",  "pill-decisions", "decisions"),
            ("actions",    "🟡 Action Items","pill-actions",   "actions"),
            ("deadlines",  "🟣 Deadlines",  "pill-deadlines", "deadlines"),
        ]

        for key, label, pill_cls, item_cls in sections:
            items = classified.get(key, [])
            if items:
                st.markdown(f'<div class="section-pill {pill_cls}">{label}</div>', unsafe_allow_html=True)
                for item in items[:4]:
                    st.markdown(f'<div class="summary-item {item_cls}">{item}.</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

        # Raw summary expander
        with st.expander("📝 View raw summary"):
            st.markdown(f'<div class="raw-summary">{summary}</div>', unsafe_allow_html=True)

        # Copy / download
        st.download_button(
            label="⬇️ Download Summary (.txt)",
            data=f"MEETING SUMMARY\n{'='*40}\n\n{summary}\n\n"
                 f"--- Structured ---\n"
                 + "\n".join(
                     f"\n{k.upper()}:\n" + "\n".join(f"  - {v}" for v in vals)
                     for k, vals in classified.items() if vals
                 ),
            file_name="meeting_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">🗂️</div>
          <p>Your structured summary will appear here<br>after you paste a transcript and click <strong>Summarize</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
