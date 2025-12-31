from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import numpy as np
import os
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="Brainv3 HITL API")

# Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMORY_PATH = os.path.join(BASE_DIR, "data", "rule_memory.json")
EXAMPLE_DIR = os.path.join(BASE_DIR, "examples", "arc_tasks")
PUZZLE_DIR = os.path.join(BASE_DIR, "puzzles")

# Models
class KnowledgeInjection(BaseModel):
    text: str
    rule_id: str = None

class ARCTask(BaseModel):
    train: List[Dict[str, Any]]
    test: List[Dict[str, Any]]

# Global Active State
ACTIVE_TASK = {
    "task_id": "arc_mirroring_01",
    "train": [],
    "test_input": None,
    "last_prediction": None,
    "current_step": 1, # 1: Load, 2: Display, 3: Response, 4: Advice, 5: Edit
}

class GridUpdate(BaseModel):
    r: int
    c: int
    color: int

# API Endpoints
@app.get("/api/memory")
async def get_memory():
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r") as f:
            return json.load(f)
    return {"rules": []}

@app.get("/api/state")
async def get_reasoning_state():
    return {
        "task_id": ACTIVE_TASK["task_id"],
        "train": ACTIVE_TASK["train"],
        "test_input": ACTIVE_TASK["test_input"],
        "predicted_grid": ACTIVE_TASK.get("last_prediction"),
        "current_step": ACTIVE_TASK["current_step"],
        "hypotheses": ACTIVE_TASK.get("active_hypotheses", []) if ACTIVE_TASK["current_step"] >= 3 else []
    }

@app.post("/api/predict")
async def trigger_predict():
    from AGI.src.swarm.predictor import ARCPredictor
    
    ACTIVE_TASK["current_step"] = 3
    ACTIVE_TASK["active_hypotheses"] = []
    
    # 1. Load Rules from Memory
    rules = []
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, dict): # Legacy format
                    rules = list(data.keys())
                else: # New schema
                    rules = [r["text"] for r in data.get("rules", [])]
        except Exception as e:
            print(f"Error loading rules: {e}")

    # 2. Find Consensus Rule
    consensus_rule = None
    train_pairs = ACTIVE_TASK.get("train", [])
    
    for rule in rules:
        is_consistent = True
        for pair in train_pairs:
            try:
                # Apply rule to training input
                pred = ARCPredictor.apply_rule(rule, pair["input"])
                # Check exact match with training output
                if pred != pair["output"]:
                    is_consistent = False
                    break
            except Exception:
                is_consistent = False
                break
        
        if is_consistent and len(train_pairs) > 0:
            consensus_rule = rule
            break
            
    # 3. Generate Prediction
    if consensus_rule and ACTIVE_TASK["test_input"]:
        print(f"Consensus Reached on Rule: {consensus_rule}")
        final_grid = ARCPredictor.apply_rule(consensus_rule, ACTIVE_TASK["test_input"])
        ACTIVE_TASK["last_prediction"] = final_grid
        ACTIVE_TASK["active_hypotheses"] = [
            {"hypothesis_id": "consensus_01", "content": consensus_rule, "score": 1.0, "evidence": ["all_train_pairs"]}
        ]
    else:
        print("No consensus rule found. Fallback to mock.")
        # Fallback (simple flip) if no rule matches
        if ACTIVE_TASK["test_input"]:
             grid = np.array(ACTIVE_TASK["test_input"])
             ACTIVE_TASK["last_prediction"] = np.flipud(grid).tolist()
             ACTIVE_TASK["active_hypotheses"] = [
                {"hypothesis_id": "fallback_01", "content": "No consistent rule found in memory.", "score": 0.0, "evidence": []}
             ]

    return {"status": "success", "step": 3, "consensus": consensus_rule}

@app.post("/api/update_test_grid")
async def update_test_grid(update: GridUpdate):
    if ACTIVE_TASK["test_input"]:
        ACTIVE_TASK["test_input"][update.r][update.c] = update.color
    return {"status": "success"}

@app.post("/api/set_step")
async def set_step(step: int):
    ACTIVE_TASK["current_step"] = step
    return {"status": "success"}

@app.get("/api/list_puzzles")
async def list_puzzles():
    if not os.path.exists(PUZZLE_DIR):
        return {"puzzles": []}
    files = [f for f in os.listdir(PUZZLE_DIR) if f.endswith(".json")]
    return {"puzzles": files}

@app.get("/api/puzzle/{filename}")
async def get_puzzle(filename: str):
    path = os.path.join(PUZZLE_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Puzzle not found")
    with open(path, "r") as f:
        return json.load(f)

@app.post("/api/upload")
async def upload_task(task: ARCTask):
    ACTIVE_TASK["train"] = task.train
    ACTIVE_TASK["test_input"] = task.test[0]["input"]
    ACTIVE_TASK["task_id"] = f"task_{os.urandom(4).hex()}"
    ACTIVE_TASK["current_step"] = 2 # Move to Display step
    ACTIVE_TASK["last_prediction"] = None
    task_storage = os.path.join(BASE_DIR, "data", "active_task.json")
    with open(task_storage, "w") as f:
        json.dump(task.model_dump(), f)
        
    return {"status": "success", "task_id": ACTIVE_TASK["task_id"]}

@app.post("/api/inject")
async def inject_knowledge(injection: KnowledgeInjection):
    print(f"Human Knowledge Injected: {injection.text}")
    # Here we would update a 'hints.json' or shared state thatSwarm reads
    hint_path = os.path.join(BASE_DIR, "data", "hints.json")
    with open(hint_path, "w") as f:
        json.dump({"hint": injection.text, "timestamp": "now"}, f)
    return {"status": "success", "message": "Knowledge injected into Swarm priority stream."}

# Serve static files for ARC images
if os.path.exists(EXAMPLE_DIR):
    app.mount("/static", StaticFiles(directory=EXAMPLE_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
