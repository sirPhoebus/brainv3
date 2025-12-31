import asyncio
from typing import List
from AGI.src.swarm.schemas import Hypothesis

class HITLInterface:
    """
    Human-in-the-Loop interface for reviewing and guiding the Swarm.
    """
    
    async def review_hypotheses(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """
        Mock HITL review. In a real system, this would prompt a user via CLI or Web.
        """
        print("\n--- HITL Review Phase ---")
        for i, h in enumerate(hypotheses[:3]): # Show top 3
            print(f"[{i}] {h.content} (Score: {h.score:.2f})")
            if h.evidence:
                print(f"    Evidence: {', '.join(h.evidence[:2])}...")
            
        # Mocking human approval: boost the first one
        if hypotheses:
            print(f"Human approved hypothesis {hypotheses[0].hypothesis_id}")
            hypotheses[0].score = min(1.0, hypotheses[0].score + 0.2)
            
        print("--- End Review ---\n")
        return hypotheses
