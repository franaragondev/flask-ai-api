from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from huggingface_hub import hf_hub_download, model_info
import threading

# -------------------------------------------------------------
# Flask App Setup
# -------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from your SvelteKit dev server

# -------------------------------------------------------------
# Global State for Models and Download Progress
# -------------------------------------------------------------

# Pipelines (lazy after warmup)
_sentiment_pipe = None
_summarize_pipe = None

# Multilingual models (explicit)
SENTIMENT_MODEL_ID = "tabularisai/multilingual-sentiment-analysis"
SUMMARIZE_MODEL_ID = "csebuetnlp/mT5_multilingual_XLSum"

# Progress structure exposed by /status
# - 'error': global error string (if any) during warmup
# - 'model_errors': per-model error messages (optional)
progress = {
    "ready": False,
    "percent": 0,
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "models": {
        "sentiment": False,
        "summarizer": False
    },
    "error": None,
    "model_errors": {
        "sentiment": None,
        "summarizer": None
    }
}

# Optional input guard for demo API
MAX_INPUT_CHARS = 8000

# -------------------------------------------------------------
# Helper: Load Pipelines
# -------------------------------------------------------------

def get_sentiment():
    """
    Return a (loaded) multilingual sentiment pipeline.
    If not initialized yet (e.g., warmup skipped), load it on demand.
    """
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_ID)
    return _sentiment_pipe

def get_summarizer():
    """
    Return a (loaded) multilingual summarization pipeline.
    If not initialized yet, load it on demand.
    """
    global _summarize_pipe
    if _summarize_pipe is None:
        _summarize_pipe = pipeline(
            "summarization",
            model=SUMMARIZE_MODEL_ID,
            tokenizer=SUMMARIZE_MODEL_ID
        )
    return _summarize_pipe

# -------------------------------------------------------------
# Warmup Download with Real(ish) Progress by Aggregated Bytes
# -------------------------------------------------------------

def _collect_total_size(model_id: str) -> int:
    """
    Get total size (in bytes) of all files in the model repo (main revision).
    Some non-LFS files may lack size metadata; we skip them (treated as 0).
    """
    try:
        info = model_info(model_id, revision="main", files_metadata=True)
        total = 0
        for f in info.siblings:
            size = getattr(f, "size", None)
            if isinstance(size, int):
                total += size
        return total
    except Exception as e:
        msg = f"failed to get model info for {model_id}: {e}"
        print(f"[warmup] {msg}")
        # Expose a global error to the frontend
        progress["error"] = msg
        return 0

def _download_model_files(model_id: str, model_key: str):
    """
    Download all files for the given model repo.
    - After each file completes, add its size to 'downloaded_bytes' and recompute 'percent'.
    - 'model_key' is 'sentiment' or 'summarizer' for per-model error reporting.
    """
    try:
        info = model_info(model_id, revision="main", files_metadata=True)
        for f in info.siblings:
            size = getattr(f, "size", None)
            try:
                # Download to local cache; hf caches by sha256 so it's idempotent/fast if already present
                hf_hub_download(
                    repo_id=model_id,
                    filename=f.rfilename,
                    revision="main",
                    local_files_only=False
                )
                # Update progress only after successful download
                if isinstance(size, int):
                    progress["downloaded_bytes"] += size
                    if progress["total_bytes"] > 0:
                        # Keep 99 as max until both pipelines are fully initialized
                        progress["percent"] = min(
                            99,
                            int(progress["downloaded_bytes"] * 100 / progress["total_bytes"])
                        )
            except Exception as ex:
                # Record per-file download failures, but continue
                print(f"[warmup] download failed {model_id}:{f.rfilename} - {ex}")
                # Also surface a per-model error (keep the last one)
                progress["model_errors"][model_key] = f"download failed: {ex}"
    except Exception as e:
        msg = f"failed to list files for {model_id}: {e}"
        print(f"[warmup] {msg}")
        progress["model_errors"][model_key] = msg

def warmup_models():
    """
    Warmup thread:
      1) Compute total bytes for both models.
      2) Download all files and update aggregated progress during download.
      3) Initialize pipelines; mark ready only when both succeed.
      4) Surface errors (global and per-model) in progress object.
    """
    global _sentiment_pipe, _summarize_pipe

    # Reset errors/progress at start
    progress["error"] = None
    progress["model_errors"]["sentiment"] = None
    progress["model_errors"]["summarizer"] = None
    progress["ready"] = False
    progress["percent"] = 0
    progress["downloaded_bytes"] = 0
    progress["total_bytes"] = 0
    progress["models"]["sentiment"] = False
    progress["models"]["summarizer"] = False

    # Step 1: compute total size (sum of both repos)
    total_sent = _collect_total_size(SENTIMENT_MODEL_ID)
    total_summ = _collect_total_size(SUMMARIZE_MODEL_ID)
    progress["total_bytes"] = total_sent + total_summ

    # Step 2: download model files (sentiment then summarizer)
    _download_model_files(SENTIMENT_MODEL_ID, "sentiment")
    try:
        _sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_ID)
        progress["models"]["sentiment"] = True
    except Exception as e:
        msg = f"sentiment init failed: {e}"
        print(f"[warmup] {msg}")
        progress["model_errors"]["sentiment"] = msg

    _download_model_files(SUMMARIZE_MODEL_ID, "summarizer")
    try:
        _summarize_pipe = pipeline("summarization", model=SUMMARIZE_MODEL_ID, tokenizer=SUMMARIZE_MODEL_ID)
        progress["models"]["summarizer"] = True
    except Exception as e:
        msg = f"summarizer init failed: {e}"
        print(f"[warmup] {msg}")
        progress["model_errors"]["summarizer"] = msg

    # Step 3: set readiness and final percent
    progress["ready"] = progress["models"]["sentiment"] and progress["models"]["summarizer"]
    if progress["ready"]:
        progress["percent"] = 100
    else:
        # If not ready, keep percent as-is (likely 99)
        # Also set a global error hint if none exists yet
        if not progress["error"]:
            progress["error"] = "warmup incomplete: check model_errors for details"

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------

@app.get("/status")
def status():
    """
    Download/Readiness status for first-time model setup.
    Frontend can poll this endpoint to render a real progress bar and show errors.
    """
    return jsonify(progress)

@app.get("/health")
def health():
    """ Basic liveness check. """
    return {"status": "ok"}

@app.post("/api/sentiment")
def sentiment():
    """
    Multilingual sentiment analysis.
    Input: { "text": "..." }
    Output: { "label": "...", "score": ... }
    """
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400
    if len(text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"text too long (>{MAX_INPUT_CHARS} chars)"}), 413

    result = get_sentiment()(text)[0]
    return jsonify(result)

@app.post("/api/summarize")
def summarize():
    """
    Multilingual summarization.
    Input: { "text": "..." }
    Output: { "summary": "..." }
    """
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400
    if len(text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"text too long (>{MAX_INPUT_CHARS} chars)"}), 413

    out = get_summarizer()(text, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
    return jsonify({"summary": out})

# -------------------------------------------------------------
# Dev Server (launch warmup thread)
# -------------------------------------------------------------

if __name__ == "__main__":
    # Start warmup in background so the server is responsive immediately
    threading.Thread(target=warmup_models, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=True)