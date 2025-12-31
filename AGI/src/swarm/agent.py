import uuid
from typing import List, Optional, Dict, Any
from AGI.src.swarm.schemas import Hypothesis, AgentAction
from AGI.src.bridge.schemas import AgentToken
from AGI.src.curiosity.scorer import CuriosityScorer

class OmnidirectionalAgent:
    """
    An agent capable of reasoning across past and future states.
    """
    
    def __init__(self, bus: Any = None, agent_id: str = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.bus = bus
        self.memory: List[AgentToken] = []
        self.current_hypotheses: List[Hypothesis] = []
        self.curiosity = CuriosityScorer()
        
    def perceive(self, tokens: List[AgentToken]):
        """
        Ingest tokens into the agent's memory.
        """
        self.memory.extend(tokens)
        
    async def propose_hypothesis(self, context: str) -> Hypothesis:
        """
        Propose a new hypothesis based on current memory and context.
        """
        h_id = f"hyp_{uuid.uuid4().hex[:8]}"
        # Calculate novelty bonus
        novelty_bonus = self.curiosity.score_hypothesis([context])
        initial_score = 0.5 + (0.1 * novelty_bonus)
        
        new_h = Hypothesis(
            hypothesis_id=h_id,
            content=f"Proposal from {self.agent_id} based on {len(self.memory)} tokens",
            score=min(1.0, initial_score),
            agent_id=self.agent_id,
            path_history=[context]
        )
        self.current_hypotheses.append(new_h)
        
        if self.bus:
            await self.bus.publish("hypotheses", new_h)
            
        return new_h

    def verify_hypothesis(self, hypothesis: Hypothesis) -> float:
        """
        Self-verify a hypothesis. Return a new score.
        'Omnidirectional' aspect: can look at 'future' predictions (mocked here).
        """
        # Mocking omnidirectional lookahead/lookback
        influence = 0.1 if "future" in hypothesis.content else -0.1
        new_score = min(1.0, max(0.0, hypothesis.score + influence))
        return new_score
