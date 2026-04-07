import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

# Import your existing service
from llm_service_vllm import RecursiveExtractionService

app = FastAPI(title="LLM Extraction UI")

# 1. Load data and service once at startup
DATA_PATH = "updated_data.json"
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        GLOBAL_DATA = json.load(f)
else:
    GLOBAL_DATA = []

# Initialize your extraction service
extraction_service = RecursiveExtractionService()

# In-memory history (For production, consider a database like SQLite)
query_history = []

class QueryRequest(BaseModel):
    question: str

# 2. Serve the static HTML frontend
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("inference.html", "r", encoding="utf-8") as f:
        return f.read()

# 3. API Endpoint to process the question
@app.post("/api/extract")
async def extract_data(request: QueryRequest):
    question = request.question
    
    # Call your existing async run function logic
    matched_items, summary = await extraction_service.extract_from_new_data(question, GLOBAL_DATA)
    
    # Save to history
    result = {
        "id": len(query_history) + 1,
        "question": question,
        "matched_items": matched_items,
        "summary": summary
    }
    query_history.insert(0, result) # Prepend to show newest first
    
    return result

# 4. API Endpoint to fetch history
@app.get("/api/history")
async def get_history():
    return query_history


# uvicorn inference_server:app --reload --port 8002