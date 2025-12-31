import numpy as np
import structlog
from typing import List

logger = structlog.get_logger()

class ARCPredictor:
    """
    Executes transformation rules on ARC grids.
    """
    
    @staticmethod
    def apply_rule(rule_content: str, input_grid: List[List[int]], demo_pair: dict = None) -> List[List[int]]:
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
            
            logger.info("applying_atomic_rule", rule=rule_lower)
                
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
            
            elif "pattern_continuation" in rule_lower:
                if "horizontal" in rule_lower:
                    current_grid = ARCPredictor.apply_pattern_continuation_horizontal(current_grid)
                elif "vertical" in rule_lower:
                    current_grid = ARCPredictor.apply_pattern_continuation_vertical(current_grid)
                else:
                    current_grid = ARCPredictor.apply_pattern_continuation_horizontal(current_grid)
                    current_grid = ARCPredictor.apply_pattern_continuation_vertical(current_grid)

            elif any(phrase in rule_lower for phrase in ["fit", "same shape", "place pattern", "insert where matches", "shape match"]):
                 if demo_pair:
                      pattern = ARCPredictor.extract_pattern_from_demo(np.array(demo_pair["input"]), np.array(demo_pair["output"]))
                      current_grid = ARCPredictor.apply_shape_fit_place(current_grid, pattern)
            
            elif "shape_fit" in rule_lower: # Explicit call for shape fit
                 if demo_pair:
                      pattern = ARCPredictor.extract_pattern_from_demo(np.array(demo_pair["input"]), np.array(demo_pair["output"]))
                      current_grid = ARCPredictor.apply_shape_fit_place(current_grid, pattern)

            else:
                logger.warning("sub_rule_not_implemented", rule=rule_lower)
                
        return current_grid.tolist()

    @staticmethod
    def apply_pattern_continuation_horizontal(grid: np.ndarray) -> np.ndarray:
        new_grid = grid.copy()
        for i in range(new_grid.shape[0]):
            row = new_grid[i]
            last_color = 0
            for j in range(new_grid.shape[1]):
                if row[j] != 0:
                    last_color = row[j]
                elif last_color != 0:
                    row[j] = last_color 
        return new_grid

    @staticmethod
    def apply_pattern_continuation_vertical(grid: np.ndarray) -> np.ndarray:
        new_grid = grid.copy()
        for j in range(new_grid.shape[1]):
            col = new_grid[:, j]
            last_color = 0
            for i in range(new_grid.shape[0]):
                if col[i] != 0:
                    last_color = col[i]
                elif last_color != 0:
                    col[i] = last_color
        return new_grid

    @staticmethod
    def extract_pattern_from_demo(demo_input: np.ndarray, demo_output: np.ndarray) -> np.ndarray:
        """Extract the 'added' pattern from demo pair (assumes no overlap/destruction)."""
        pattern = demo_output.copy()
        try:
             # Ensure shapes match for subtraction
             if demo_input.shape == demo_output.shape:
                 # Capture differences: Keep pixels that are DIFFERENT in output vs input
                 # This handles both "added objects" and "color changes"
                 pattern[demo_input == demo_output] = 0
        except Exception:
             pass # Fallback to using output as pattern if shapes differ drastically
        
        logger.info("extracted_pattern", shape=pattern.shape, non_zero=np.count_nonzero(pattern))
        return pattern

    @staticmethod
    def find_fit_locations(test_input: np.ndarray, pattern: np.ndarray) -> List[tuple]:
        """Find top-left positions where pattern can be placed without overlapping existing colors."""
        from scipy.ndimage import label, find_objects
        
        if np.all(pattern == 0):
            return []
        
        # Get bounding box of the pattern object (assume one main connected component)
        labeled, num_features = label(pattern != 0)
        if num_features == 0:
            return []
        
        try:
            obj_slice = find_objects(labeled)[0]  # Take first/main object
            obj = pattern[obj_slice]
            ph, pw = obj.shape
            
            locations = []
            ih, iw = test_input.shape
            for y in range(ih - ph + 1):
                for x in range(iw - pw + 1):
                    slice_grid = test_input[y:y+ph, x:x+pw]
                    # Fit condition: where pattern has color, test must be empty (0)
                    if np.all(slice_grid[obj != 0] == 0):
                        locations.append((y - obj_slice[0].start, x - obj_slice[1].start))  # Adjust for bounding
            return locations
        except Exception:
            return []

    @staticmethod
    def apply_shape_fit_place(test_input: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """Place the pattern in the first valid fit location."""
        predicted = test_input.copy()
        locations = ARCPredictor.find_fit_locations(test_input, pattern)
        logger.info("shape_fit_locations_found", count=len(locations), first_location=locations[0] if locations else None)
        if not locations:
            return predicted  # No fit, return unchanged
        
        # Place in first valid location
        y_offset, x_offset = 0, 0
        from scipy.ndimage import label, find_objects
        
        labeled, _ = label(pattern != 0)
        obj_slice = find_objects(labeled)[0]
        obj = pattern[obj_slice]
        ph, pw = obj.shape
        
        # locations[0] is the top-left of the bounding box relative to test grid
        # We need to map the full pattern onto test grid using this offset
        # Actually, find_fit_locations returns the top-left of where the *original pattern array* should start
        # to align the object correctly.
        
        y_orig, x_orig = locations[0]
        # Copy non-zero pixels
        for r in range(pattern.shape[0]):
            for c in range(pattern.shape[1]):
                 if pattern[r, c] != 0:
                     if 0 <= y_orig + r < predicted.shape[0] and 0 <= x_orig + c < predicted.shape[1]:
                          predicted[y_orig + r, x_orig + c] = pattern[r, c]
                          
        return predicted
