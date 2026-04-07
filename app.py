import json
import torch
import uvicorn
import threading
from typing import Dict, Optional
from pydantic import BaseModel
from unsloth import FastLanguageModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

MODEL_PATH = "Qwen/Qwen3-4B-Instruct-2507"

MAX_SEQ_LENGTH = 2048

model = None

tokenizer = None

gpu_lock = threading.Lock() 

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model, tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(model_name = MODEL_PATH, max_seq_length = MAX_SEQ_LENGTH, dtype = None, load_in_4bit = True, device_map = "cuda")
    
    FastLanguageModel.for_inference(model)
    
    print("Model loaded successfully")

    yield

    del model

    del tokenizer

    torch.cuda.empty_cache()

app = FastAPI(title="Semantic Keyword Extractor API", lifespan=lifespan)

class ExtractionRequest(BaseModel):

    passage: str

    glossary: Dict[str, str]

    system_prompt: Optional[str] = None

@app.post("/extract")
def extract_keywords(data: ExtractionRequest):
    
    if model is None or tokenizer is None:
    
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:

        formatted_glossary_lines = [f"- {k}: {v}" for k, v in data.glossary.items()]

        glossary_str = "\n".join(formatted_glossary_lines)
        
        messages = [
            {
                "role": "system",
                "content": data.system_prompt,
            },
            {
                "role": "user",
                "content": f"Text passage:\n{data.passage}\n\nGlossary:\n{glossary_str}",
            },
        ]

        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")

        # The lock ensures only ONE thread can execute this block at a time
        with gpu_lock: 
            
            with torch.inference_mode():
                
                outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], 
                                         max_new_tokens=150, use_cache=True, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
        
        generated_text = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
       
        cleaned_output = generated_text.strip()
        
        try:
        
            return json.loads(cleaned_output)
        
        except json.JSONDecodeError:
        
            return {"raw_output": cleaned_output}

    except Exception as e:
        
        print(f"Error: {e}")
        
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)