import json
import uvicorn
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

DATA_PATH = "updated_data.json"

app = FastAPI()


def load():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save(data):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@app.get("/data")
def get_data():
    return load()


class UpdateReq(BaseModel):
    # Path encoding:
    #   [ei]                        → top-level entity at index ei
    #   [ei, gi, ii]                → entity's child at group gi, item ii
    #   [ei, gi, ii, gi2, ii2]      → that child's child at group gi2, item ii2
    #   … and so on (pairs of group_idx, item_idx after the first element)
    path: List[int]
    description: str
    short_description: str = ""   # only meaningful for node items


@app.put("/update")
def update(req: UpdateReq):
    data = load()
    path = req.path

    if not path:
        raise HTTPException(status_code=400, detail="Empty path")

    try:
        # Start at the top-level entity
        item = data[path[0]]
        # Navigate through each subsequent (group_idx, item_idx) pair
        for k in range(1, len(path), 2):
            gi, ii = path[k], path[k + 1]
            item = item["children"][gi][ii]
        item["description"] = req.description
        # Only persist short_description on node items
        if "short_description" in item:
            item["short_description"] = req.short_description
    except (IndexError, KeyError, TypeError):
        raise HTTPException(status_code=404, detail="Item not found at given path")

    save(data)
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def editor_page():
    with open("editor.html", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)