import json
import uvicorn
from typing import Dict, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from llama_cpp import Llama

# Point this to the GGUF file you generated in Step 1
MODEL_PATH = "qwen3-4b-instruct-unsloth.Q4_K_M.gguf"
MAX_SEQ_LENGTH = 2048

llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    
    # n_gpu_layers=-1 offloads ALL layers to the GPU
    # n_ctx sets the context window
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1, 
        n_ctx=MAX_SEQ_LENGTH,
        verbose=False # Set to True for debugging
    )
    
    print("GGUF Model loaded successfully")
    yield
    del llm

app = FastAPI(title="Semantic Keyword Extractor API", lifespan=lifespan)

class ExtractionRequest(BaseModel):
    passage: str
    glossary: Dict[str, str]
    system_prompt: Optional[str] = None

@app.post("/extract")
def extract_keywords(data: ExtractionRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        formatted_glossary_lines = [f"- {k}: {v}" for k, v in data.glossary.items()]
        glossary_str = "\n".join(formatted_glossary_lines)
        
        messages = [
            {"role": "system", "content": data.system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": f"Text passage:\n{data.passage}\n\nGlossary:\n{glossary_str}"}
        ]

        # llama-cpp-python handles the inference efficiently without manual locks
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.1,
            response_format={"type": "json_object"} # Forces JSON output if the model supports it
        )
        
        generated_text = response["choices"][0]["message"]["content"].strip()
        
        try:
            return json.loads(generated_text)
        except json.JSONDecodeError:
            return {"raw_output": generated_text}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)