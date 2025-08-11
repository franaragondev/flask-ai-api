from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

# Create a Flask application instance
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) for development
# This allows requests from your frontend (e.g., SvelteKit running on localhost:5173)
CORS(app)

# Lazy-loaded model variables (initialized only when first needed)
# This saves time during startup and only loads models into memory when an endpoint is called
_sentiment_pipe = None

def get_sentiment():
    """
    Load (if not already loaded) and return the sentiment analysis pipeline.
    - This uses a pre-trained model from Hugging Face to classify text as POSITIVE/NEGATIVE/NEUTRAL.
    """
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline("sentiment-analysis")  # Default: DistilBERT-based model
    return _sentiment_pipe

@app.get("/health")
def health():
    """
    Health check endpoint.
    - Returns a simple JSON object to confirm the API is running.
    - Can be used by monitoring tools or the frontend to verify the backend is alive.
    """
    return {"status": "ok"}

@app.post("/api/sentiment")
def sentiment():
    """
    Sentiment analysis endpoint.
    - Expects JSON: { "text": "some text here" }
    - Returns a JSON object with label (POSITIVE/NEGATIVE/NEUTRAL) and score (confidence level).
    """
    data = request.get_json() or {}  # Get JSON body from request, or empty dict if missing
    text = (data.get("text") or "").strip()  # Extract 'text' and remove extra spaces
    if not text:
        # If 'text' is missing or empty, return HTTP 400 Bad Request
        return jsonify({"error": "text is required"}), 400
    
    # Run the sentiment model on the input text
    # Example output: {'label': 'POSITIVE', 'score': 0.987}
    out = get_sentiment()(text)[0]
    return jsonify(out)  # Send the result as JSON