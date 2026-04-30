# Smart File AI Chatbot

Smart File AI Chatbot is a local document chat app built with FastAPI, Ollama, ChromaDB, and a lightweight browser UI. You can upload PDF, DOCX, or TXT files, then ask questions against them without sending data to an external API.

## What it does

- Upload documents and turn them into searchable context
- Ask questions with or without RAG
- Use a local Ollama model for answers and embeddings
- See timing details for retrieval and generation
- Run everything on your own machine

## Project layout

- `backend/` - API, RAG pipeline, file handling, vector search, logging
- `frontend/` - Browser chat interface
- `data/` - Local app data and vector storage
- `logs/` - JSON chat logs
- `run_all.py` and `start_*.bat/.sh` - Startup helpers

## Requirements

- Python 3.9 or newer
- Ollama installed and running
- Enough disk space for the model and local index

## Quick start

1. Install Ollama from https://ollama.com and pull a model such as `llama2` or `mistral`.
2. Set up the backend:

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Start the backend:

```bash
python main.py
```

4. Open the frontend in your browser:

```text
frontend/index.html
```

If you prefer serving it locally, run `python -m http.server 8080 --directory frontend` and open `http://localhost:8080`.

## Using the app

Upload a document, wait for it to finish processing, then ask a question in the chat box. Keep RAG on when you want answers grounded in your files. Turn it off for general questions that do not need document context.

Each response shows timing details so you can see how long embedding, retrieval, and generation took.

## API quick reference

- `GET /health`
- `POST /upload`
- `POST /query`
- `GET /statistics`
- `GET /models`
- `POST /clear-database`

## Notes

- Local runtime data and logs are ignored by git.
- The app is designed to work offline after setup.