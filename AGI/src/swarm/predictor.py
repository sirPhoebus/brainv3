import numpy as np
import structlog
from typing import List

logger = structlog.get_logger()

class ARCPredictor:
    """
    Executes transformation rules on ARC grids.
    """
    
    @staticmethod
    def apply_rule(rule_content: str, input_grid: List[List[int]]) -> List[List[int]]:
        """
        Apply a textual rule (or chain of rules) to a grid.
        Rules can be separated by commas or 'including'.
        """
        import re
        # Normalize and split into atomic rules
        atomic_rules = re.split(r',| including ', rule_content.lower())
        
        current_grid = np.array(input_grid)
        
        for rule_raw in atomic_rules:
            rule_lower = rule_raw.strip()
            if not rule_lower:
                continue
                
            # Logic for each atomic rule
            if "identity" in rule_lower:
                pass # stays same
                
            elif "reflection" in rule_lower:
                if "top" in rule_lower and "bottom" in rule_lower:
                    mid = current_grid.shape[0] // 2
                    res = current_grid.copy()
                    res[mid:] = np.flipud(current_grid[:mid])
                    current_grid = res
                elif "left" in rule_lower and "right" in rule_lower:
                    mid = current_grid.shape[1] // 2
                    res = current_grid.copy()
                    res[:, mid:] = np.fliplr(current_grid[:, :mid])
                    current_grid = res

            elif "color_fill" in rule_lower:
                vals, counts = np.unique(current_grid[current_grid > 0], return_counts=True)
                if len(vals) > 0:
                    most_freq = vals[np.argmax(counts)]
                    res = current_grid.copy()
                    res[res == 0] = most_freq
                    current_grid = res
                    current_grid = res
            
            elif "rotation" in rule_lower:
                k = 0
                if "90" in rule_lower: k = 1 # 90 deg clockwise (numpy rot90 is counter-clockwise, so we use k=-1 or 3)
                elif "180" in rule_lower: k = 2
                elif "270" in rule_lower: k = 3
                
                # np.rot90 rotates counter-clockwise by default. 
                # k=1 -> 90deg CCW. 
                # Requirement: "90 degrees clockwise" -> 270 degrees CCW -> k=3 (or k=-1)
                if k == 1: k = 3 
                elif k == 3: k = 1
                
                current_grid = np.rot90(current_grid, k=k)
            
            else:
                logger.warning("sub_rule_not_implemented", rule=rule_lower)
                
        return current_grid.tolist()
