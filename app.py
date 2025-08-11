from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

# -------------------------------------------------------------
# Flask App Setup
# -------------------------------------------------------------

# Create a Flask application instance (this is the web server)
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) for development
# This is necessary so that your frontend running on a different port
# (e.g., SvelteKit on localhost:5173) can make requests to this backend
CORS(app)

# -------------------------------------------------------------
# Lazy-Loaded Models
# -------------------------------------------------------------

# These variables will store our loaded AI models.
# We keep them as None until they are actually used, to save startup time.
_sentiment_pipe = None
_summarize_pipe = None

# Optional guard to avoid extremely large input texts in demo usage
MAX_INPUT_CHARS = 8000  # You can increase this if you want

# -------------------------------------------------------------
# Helper Functions to Load Models
# -------------------------------------------------------------

def get_sentiment():
    """
    Load and return the multilingual sentiment analysis pipeline.
    - This model can understand multiple languages (EN, ES, FR, etc.)
    - The pipeline object is stored globally so it's loaded only once.
    """
    global _sentiment_pipe
    if _sentiment_pipe is None:
        # 'sentiment-analysis' task will use the given multilingual model
        _sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="tabularisai/multilingual-sentiment-analysis"
        )
    return _sentiment_pipe

def get_summarizer():
    """
    Load and return the multilingual summarization pipeline.
    - This uses an mT5-based model trained on the XLSum dataset.
    - The tokenizer is specified to match the model.
    - Summarization works best on moderate-length inputs (few paragraphs).
    """
    global _summarize_pipe
    if _summarize_pipe is None:
        _summarize_pipe = pipeline(
            "summarization",
            model="csebuetnlp/mT5_multilingual_XLSum",
            tokenizer="csebuetnlp/mT5_multilingual_XLSum"
        )
    return _summarize_pipe

# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------

@app.get("/health")
def health():
    """
    Health check endpoint.
    - Use this to verify that the backend is running.
    - Returns {"status": "ok"} if the server is alive.
    - Can be called from the frontend before sending AI requests.
    """
    return {"status": "ok"}

@app.post("/api/sentiment")
def sentiment():
    """
    Multilingual Sentiment Analysis Endpoint.
    - Receives JSON: { "text": "..." }
    - Returns JSON: { "label": "...", "score": ... }
    - 'label' is POSITIVE / NEGATIVE / NEUTRAL (depending on model output)
    - 'score' is the confidence value between 0 and 1.
    """
    # Parse JSON request body, fallback to empty dict if invalid/missing
    data = request.get_json() or {}
    
    # Extract 'text' field and remove leading/trailing spaces
    text = (data.get("text") or "").strip()

    # Validate the text input
    if not text:
        return jsonify({"error": "text is required"}), 400
    if len(text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"text too long (>{MAX_INPUT_CHARS} chars)"}), 413

    # Perform sentiment analysis using our loaded pipeline
    # The pipeline returns a list of results; we take the first one
    result = get_sentiment()(text)[0]

    # Send the result back as JSON
    return jsonify(result)

@app.post("/api/summarize")
def summarize():
    """
    Multilingual Text Summarization Endpoint.
    - Receives JSON: { "text": "..." }
    - Returns JSON: { "summary": "..." }
    - Summarization will produce a shorter text that retains the main idea.
    """
    # Parse JSON request body
    data = request.get_json() or {}
    
    # Extract 'text' field and remove spaces
    text = (data.get("text") or "").strip()

    # Validate input
    if not text:
        return jsonify({"error": "text is required"}), 400
    if len(text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"text too long (>{MAX_INPUT_CHARS} chars)"}), 413

    # Run summarization model with length constraints
    # - max_length: maximum number of tokens in output
    # - min_length: minimum number of tokens in output
    # - do_sample=False: deterministic output
    out = get_summarizer()(text, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]

    # Send JSON response with the summary
    return jsonify({"summary": out})

# -------------------------------------------------------------
# Development Server
# -------------------------------------------------------------

if __name__ == "__main__":
    """
    Run the Flask development server.
    - host="127.0.0.1": only accessible locally
    - port=5000: API available at http://127.0.0.1:5000
    - debug=True: enables auto-reload and detailed error messages
    """
    app.run(host="127.0.0.1", port=5000, debug=True)