# Flask AI API

AI-powered Flask backend providing sentiment analysis and text summarization endpoints,
designed to integrate with the SvelteKit AI TextLab frontend.

## Endpoints

- GET `/health` – service health
- POST `/api/sentiment` – `{ text }` -> `{ label, score }`
- POST `/api/summarize` – `{ text }` -> `{ summary }`

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
