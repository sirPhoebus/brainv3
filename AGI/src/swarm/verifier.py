from typing import List
from AGI.src.swarm.schemas import Hypothesis
import structlog

logger = structlog.get_logger()

class SwarmVerifier:
    """
    Independent module to verify the consistency and quality of swarm outputs.
    """
    
    @staticmethod
    def verify_consistency(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Step 4: Prune/Strengthen.
        Global check for contradictions and consensus.
        """
        if not hypotheses:
            return []
            
        # Strengthen hypotheses with many cross-validation markers
        for h in hypotheses:
            cross_val_count = sum(1 for e in h.evidence if "Cross-validated" in e)
            if cross_val_count > 0:
                h.score = min(1.0, h.score + (0.05 * cross_val_count))
                logger.debug("verifier_strengthen", h_id=h.hypothesis_id, cross_vals=cross_val_count, new_score=h.score)
                
        # Weaken hypotheses with no evidence or very low agent support
        for h in hypotheses:
            if not h.evidence and h.score < 0.6:
                h.score -= 0.1
                
        return hypotheses

    @staticmethod
    def merge_similar(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Merge hypotheses that describe similar scene elements to prevent redundancy.
        """
        if len(hypotheses) < 2:
            return hypotheses
            
        merged = []
        seen_content = set()
        
        # Sort by score so we keep the best versions
        sorted_h = sorted(hypotheses, key=lambda x: x.score, reverse=True)
        
        for h in sorted_h:
            # Simple keyword-based similarity for prototype
            content_key = h.content.split("detects")[-1].strip() if "detects" in h.content else h.content
            if content_key in seen_content:
                # Strengthen the existing one instead of adding new
                for m in merged:
                    if content_key in m.content:
                        m.score = min(1.0, m.score + 0.02)
                        m.evidence.append(f"Merged with redundant hypothesis from {h.agent_id[:4]}")
                continue
            
            merged.append(h)
            seen_content.add(content_key)
            
        return merged

    @staticmethod
    def prune_conflicts(hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Remove hypotheses that are logically inconsistent (mocked).
        """
        # Remove anything with a very low score
        return [h for h in hypotheses if h.score > 0.2]
