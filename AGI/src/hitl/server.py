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
    human_grid: List[List[int]] = None

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
    
    # Log current state for debugging
    test_grid = ACTIVE_TASK.get("test_input")
    if test_grid:
        test_array = np.array(test_grid)
        print(f"PREDICT: Using test_input with shape {test_array.shape}, hash={hash(str(test_grid))}, sample={test_grid[0][:5] if test_grid else 'None'}")
    
    ACTIVE_TASK["current_step"] = 3
    ACTIVE_TASK["active_hypotheses"] = []
    
    # 1. Load Rules from Memory
    rules = []
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r") as f:
                data = json.load(f)
            with open(MEMORY_PATH, "r") as f:
                data = json.load(f)
                if "rules" in data and isinstance(data["rules"], list): 
                    # New Schema: {"rules": [{"text":...}, ...]}
                    rules = [r["text"] for r in data["rules"]]
                elif isinstance(data, dict): 
                    # Legacy Schema: {"rule_text": weight}
                    rules = list(data.keys())
        except Exception as e:
            print(f"Error loading rules: {e}")

     # 1.5 Load Hints from hints.json (High Priority)
    HINTS_PATH = os.path.join(BASE_DIR, "data", "hints.json")
    if os.path.exists(HINTS_PATH):
        try:
            with open(HINTS_PATH, "r") as f:
                hint_data = json.load(f)
                hint_text = hint_data.get("hint")
                human_grid = hint_data.get("solution") or hint_data.get("correct_grid")
                
                # IMMEDIATE TRUTH OVERRIDE
                if human_grid:
                    print("Found Human Truth in hints.json - OVERRIDING PREDICTION")
                    ACTIVE_TASK["last_prediction"] = human_grid
                    ACTIVE_TASK["active_hypotheses"] = [{
                        "hypothesis_id": "human_truth_01", 
                        "content": f"Perfect Solution provided by Human Feedback: {hint_text or 'Direct Grid'}", 
                        "score": 1.0, 
                        "evidence": ["human_feedback", "ground_truth", "hints_file"]
                    }]
                    return {"status": "success", "step": 3, "consensus": "Human Truth"}

                if hint_text:
                     print(f"Loaded Hint as Rule: {hint_text}")
                     # Prepend to check it first
                     rules.insert(0, hint_text)
        except Exception as e:
             print(f"Error loading hints: {e}")

    # 2. Prioritize Human Verified Consensus
    consensus_rule = None
    train_pairs = ACTIVE_TASK.get("train", [])
    human_sol = ACTIVE_TASK.get("human_solution")

    if human_sol:
        print("Validating rules against Human Truth...")
        for rule in rules:
            try:
                # Check if rule validates against Human Solution (Test Input -> Human Output)
                # We still use train_pairs[0] as the pattern source
                pred = ARCPredictor.apply_rule(rule, ACTIVE_TASK["test_input"], demo_pair=train_pairs[0])
                if pred == human_sol:
                    print(f"Rule verified by Human Solution: {rule}")
                    consensus_rule = rule
                    # Optionally we could also verify against train_pairs here to serve as "Sanity Check"
                    # But human intent overrides all.
                    break
            except Exception as e:
                continue
    
    # 2b. Standard Consensus (if no human solution or no rule matched it)
    if not consensus_rule:
        for rule in rules:
            is_consistent = True
            for pair in train_pairs:
                try:
                    # Apply rule to training input. 
                    # We use the FIRST training example as the 'source' of the pattern to be consistent.
                    pred = ARCPredictor.apply_rule(rule, pair["input"], demo_pair=train_pairs[0])
                    
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
        print(f"Applying to test_input: shape={np.array(ACTIVE_TASK['test_input']).shape}")
        final_grid = ARCPredictor.apply_rule(consensus_rule, ACTIVE_TASK["test_input"], demo_pair=train_pairs[0])
        ACTIVE_TASK["last_prediction"] = final_grid
        ACTIVE_TASK["active_hypotheses"] = [
            {"hypothesis_id": "consensus_01", "content": consensus_rule, "score": 1.0, "evidence": ["all_train_pairs"]}
        ]
    else:
        # Check if we have a hint that we can force-execute
        hint_applied = False
        HINTS_PATH = os.path.join(BASE_DIR, "data", "hints.json")
        if os.path.exists(HINTS_PATH) and ACTIVE_TASK["test_input"]:
             try:
                with open(HINTS_PATH, "r") as f:
                    hint_data = json.load(f)
                    hint_text = hint_data.get("hint")
                    if hint_text:
                        print(f"Force-executing Hint: {hint_text}")
                        # Apply hint to test input
                        final_grid = ARCPredictor.apply_rule(hint_text, ACTIVE_TASK["test_input"], demo_pair=train_pairs[0] if train_pairs else None)
                        
                        # Only use hint result if it actually changed the grid
                        if final_grid != ACTIVE_TASK["test_input"]:
                            ACTIVE_TASK["last_prediction"] = final_grid
                            ACTIVE_TASK["active_hypotheses"] = [
                                {"hypothesis_id": "hint_force_01", "content": f"[Hint] {hint_text}", "score": 0.5, "evidence": ["user_hint_only"]}
                            ]
                            hint_applied = True
                            print(f"Hint produced a valid transformation.")
                        else:
                            print(f"WARNING: Hint did not transform the grid. Likely causes: Pattern extraction failed (no objects found) or no valid fit locations.")
             except Exception as e:
                 print(f"Failed to force hint: {e}")

        if not hint_applied:
            print("No consensus rule found. Fallback to mock.")
            # Fallback (simple flip) if no rule matches
            if ACTIVE_TASK["test_input"]:
                 grid = np.array(ACTIVE_TASK["test_input"])
                 ACTIVE_TASK["last_prediction"] = np.flipud(grid).tolist()
                 ACTIVE_TASK["active_hypotheses"] = [
                    {"hypothesis_id": "fallback_01", "content": "No consistent rule found in memory.", "score": 0.0, "evidence": []}
                 ]
                 
    # 4. Inject Human Truth if available
    human_sol = ACTIVE_TASK.get("human_solution")
    if human_sol:
        ACTIVE_TASK["active_hypotheses"].insert(0, {
            "hypothesis_id": "human_truth_01", 
            "content": "Perfect Solution provided by Human Feedback.", 
            "score": 1.0, 
            "evidence": ["human_feedback", "ground_truth"]
        })
        # Override prediction visual
        ACTIVE_TASK["last_prediction"] = human_sol

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
    
    if injection.human_grid:
        print("Human solution grid received.")
        ACTIVE_TASK["human_solution"] = injection.human_grid
        
    # Here we would update a 'hints.json' or shared state thatSwarm reads
    hint_path = os.path.join(BASE_DIR, "data", "hints.json")
    with open(hint_path, "w") as f:
        data = {"hint": injection.text, "timestamp": "now"}
        if injection.human_grid:
            data["solution"] = injection.human_grid
        json.dump(data, f)
        
    return {"status": "success", "message": "Knowledge and solution injected."}

# Serve static files for ARC images
if os.path.exists(EXAMPLE_DIR):
    app.mount("/static", StaticFiles(directory=EXAMPLE_DIR), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
