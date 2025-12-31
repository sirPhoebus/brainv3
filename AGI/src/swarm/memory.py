import json
import os
import structlog
from typing import List, Dict

logger = structlog.get_logger()

class RuleMemory:
    """
    Persistent storage for successful ARC transformation rules.
    """
    def __init__(self, storage_path: str = "AGI/data/rule_memory.json"):
        self.storage_path = storage_path
        self.rules: Dict[str, float] = {} # Rule -> Global Confidence/Success Count
        self._load()
        
    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                try:
                    self.rules = json.load(f)
                except:
                    self.rules = {}
                    
    def persist_rule(self, rule: str, weight: float = 1.0):
        """
        Record a successful rule.
        """
        if rule in self.rules:
            self.rules[rule] += weight
        else:
            self.rules[rule] = weight
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.rules, f, indent=4)
        logger.info("rule_persisted", rule=rule, current_weight=self.rules[rule])

    def get_top_rules(self, k: int = 5) -> List[str]:
        """
        Retrieve successful rules to bias future reasoning.
        """
        sorted_rules = sorted(self.rules.items(), key=lambda x: x[1], reverse=True)
        return [r[0] for r in sorted_rules[:k]]
