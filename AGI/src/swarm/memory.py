import json
import os
import structlog
import random
from datetime import datetime
from typing import List, Dict, Optional

logger = structlog.get_logger()

class RuleMemory:
    """
    Persistent storage for successful ARC transformation rules with no forgetting.
    Implements weighted rules, rehearsal mechanism, and decay.
    """
    def __init__(self, storage_path: str = "AGI/data/rule_memory.json"):
        self.storage_path = storage_path
        self.rules: List[Dict] = []
        self._load()
        
    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                try:
                    data = json.load(f)
                    # Support legacy dict format if it exists, or the new list format
                    if isinstance(data, dict):
                        # Convert legacy Dict[str, float] to List[Dict]
                        self.rules = [
                            {
                                "text": text,
                                "weight": weight,
                                "success_count": int(weight * 2), # Heuristic conversion
                                "last_used": datetime.now().isoformat()
                            }
                            for text, weight in data.items()
                        ]
                    else:
                        self.rules = data.get("rules", [])
                except Exception as e:
                    logger.error("memory_load_failed", error=str(e))
                except Exception as e:
                    logger.error("memory_load_failed", error=str(e))
                    self.rules = []

        # Load hints as high priority rules
        hints_path = os.path.join(os.path.dirname(self.storage_path), "hints.json")
        if os.path.exists(hints_path):
             try:
                with open(hints_path, 'r') as f:
                    data = json.load(f)
                    # hints.json usually has {"hint": "text", "timestamp": "..."}
                    # We treat it as a single high-priority rule injection
                    hint_text = data.get("hint")
                    if hint_text:
                        # Check if already exists to update or append
                        for rule in self.rules:
                            if rule["text"] == hint_text:
                                rule["weight"] = 3.0 # Super high priority
                                break
                        else:
                            self.rules.append({
                                "text": hint_text,
                                "weight": 3.0,
                                "success_count": 0,
                                "last_used": datetime.now().isoformat()
                            })
             except Exception as e:
                logger.error("hints_load_failed", error=str(e))
                    
    def save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump({"rules": self.rules}, f, indent=4)
            
    def add_or_update(self, rule_text: str):
        """
        Record a successful rule. Increases weight and success count.
        """
        for rule in self.rules:
            if rule["text"] == rule_text:
                rule["success_count"] += 1
                rule["weight"] = min(2.0, rule.get("weight", 1.0) + 0.5)
                rule["last_used"] = datetime.now().isoformat()
                break
        else:
            self.rules.append({
                "text": rule_text,
                "weight": 1.0,
                "success_count": 1,
                "last_used": datetime.now().isoformat()
            })
        self.save()
        logger.info("memory_updated", rule=rule_text)

    def decay_unused(self, used_this_run: List[str]):
        """
        Light decay for rules not used in the current run.
        """
        for rule in self.rules:
            if rule["text"] not in used_this_run:
                rule["weight"] = max(0.3, rule.get("weight", 1.0) - 0.01)
        self.save()
        logger.info("memory_decayed")

    def get_weighted_rules(self, top_n: Optional[int] = None) -> List[Dict]:
        """
        Retrieve rules sorted by weight.
        """
        sorted_rules = sorted(self.rules, key=lambda x: x.get("weight", 1.0), reverse=True)
        return sorted_rules[:top_n] if top_n else sorted_rules

    def get_rehearsal_candidates(self, n: int = 3) -> List[Dict]:
        """
        Lowest weight rules but still above floor.
        """
        low_weight = [r for r in self.rules if r.get("weight", 1.0) < 0.8]
        if not low_weight:
            return []
        return random.sample(low_weight, min(n, len(low_weight)))

    # Legacy compatibility
    def persist_rule(self, rule: str, weight: float = 1.0):
        self.add_or_update(rule)

    def get_top_rules(self, k: int = 5) -> List[str]:
        return [r["text"] for r in self.get_weighted_rules(top_n=k)]
