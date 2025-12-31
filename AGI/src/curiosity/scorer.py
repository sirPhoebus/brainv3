from typing import Set, List
import math

class CuriosityScorer:
    """
    Evaluates novelty and rewards exploration.
    """
    
    def __init__(self):
        self.visited_states: Set[str] = set()
        
    def calculate_novelty(self, state_representation: str) -> float:
        """
        Calculate novelty score based on frequency of visitation.
        Higher score means more novel.
        """
        if state_representation not in self.visited_states:
            self.visited_states.add(state_representation)
            return 1.0
        
        # Simple decay or fixed low reward for repeated states
        return 0.1
    
    def score_hypothesis(self, path: List[str]) -> float:
        """
        Score a reasoning path based on the novelty of its steps.
        """
        novelty_sum = sum(self.calculate_novelty(step) for step in path)
        if not path:
            return 0.0
        return novelty_sum / len(path)
