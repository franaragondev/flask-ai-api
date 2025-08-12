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
# Global State (only sentiment now)
# -------------------------------------------------------------
_sentiment_pipe = None

SENTIMENT_MODEL_ID = "tabularisai/multilingual-sentiment-analysis"

progress = {
    "ready": False,
    "percent": 0,
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "models": {
        "sentiment": False
    },
    "error": None,
    "model_errors": {
        "sentiment": None
    }
}

MAX_INPUT_CHARS = 8000

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def get_sentiment():
    """Return a (loaded) multilingual sentiment pipeline."""
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_ID)
    return _sentiment_pipe

def _collect_total_size(model_id: str) -> int:
    """Sum file sizes for progress bar."""
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
        progress["error"] = msg
        return 0

def _download_model_files(model_id: str, model_key: str):
    """Download all model files and update progress."""
    try:
        info = model_info(model_id, revision="main", files_metadata=True)
        for f in info.siblings:
            size = getattr(f, "size", None)
            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=f.rfilename,
                    revision="main",
                    local_files_only=False
                )
                if isinstance(size, int):
                    progress["downloaded_bytes"] += size
                    if progress["total_bytes"] > 0:
                        progress["percent"] = min(
                            99, int(progress["downloaded_bytes"] * 100 / progress["total_bytes"])
                        )
            except Exception as ex:
                print(f"[warmup] download failed {model_id}:{f.rfilename} - {ex}")
                progress["model_errors"][model_key] = f"download failed: {ex}"
    except Exception as e:
        msg = f"failed to list files for {model_id}: {e}"
        print(f"[warmup] {msg}")
        progress["model_errors"][model_key] = msg

def warmup_models():
    """Download and initialize sentiment model once."""
    global _sentiment_pipe
    progress["error"] = None
    progress["model_errors"]["sentiment"] = None
    progress["ready"] = False
    progress["percent"] = 0
    progress["downloaded_bytes"] = 0
    progress["total_bytes"] = 0
    progress["models"]["sentiment"] = False

    total = _collect_total_size(SENTIMENT_MODEL_ID)
    progress["total_bytes"] = total

    _download_model_files(SENTIMENT_MODEL_ID, "sentiment")
    try:
        _sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_ID)
        progress["models"]["sentiment"] = True
    except Exception as e:
        msg = f"sentiment init failed: {e}"
        print(f"[warmup] {msg}")
        progress["model_errors"]["sentiment"] = msg

    progress["ready"] = progress["models"]["sentiment"]
    progress["percent"] = 100 if progress["ready"] else progress["percent"]  # keep last if failed

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/status")
def status():
    """Readiness/progress for first-time model setup."""
    return jsonify(progress)

@app.get("/health")
def health():
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

# -------------------------------------------------------------
# Dev Server
# -------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=warmup_models, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=True)