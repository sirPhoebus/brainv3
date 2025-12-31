import uuid
import structlog
from typing import List, Optional, Dict, Any
from AGI.src.swarm.schemas import Hypothesis, AgentAction
from AGI.src.bridge.schemas import AgentToken
from AGI.src.curiosity.scorer import CuriosityScorer
from AGI.src.config_loader import DEFAULT_CONFIG

logger = structlog.get_logger()

class OmnidirectionalAgent:
    """
    An agent capable of reasoning across past and future states.
    """
    
    def __init__(self, bus: Any = None, agent_id: str = None):
        self.config = DEFAULT_CONFIG.get("curiosity", {})
        self.agent_id = agent_id or str(uuid.uuid4())
        self.bus = bus
        self.memory: List[AgentToken] = []
        self.current_hypotheses: List[Hypothesis] = []
        self.curiosity = CuriosityScorer()
        
    def perceive(self, tokens: List[AgentToken]):
        """
        Ingest tokens into the agent's memory.
        """
        logger.debug("agent_perceive", agent_id=self.agent_id, num_tokens=len(tokens))
        self.memory.extend(tokens)
        
    async def propose_hypothesis(self, context: str) -> Hypothesis:
        """
        Propose a new hypothesis based on current memory and context.
        """
        h_id = f"hyp_{uuid.uuid4().hex[:8]}"
        # Calculate novelty bonus
        novelty_bonus = self.curiosity.score_hypothesis([context])
        weight = self.config.get("novelty_bonus_weight", 0.1)
        base = self.config.get("base_score", 0.5)
        initial_score = base + (weight * novelty_bonus)
        
        # Use metadata to enrich hypothesis if available
        context_str = context
        if self.memory:
            top_token = self.memory[-1]
            # Metadata is in top_token if available (passed from VisualSegment)
            # For simplicity, we just note the token count
            pass
            
        new_h = Hypothesis(
            hypothesis_id=h_id,
            content=f"Detection Proposal from {self.agent_id} analyzing {len(self.memory)} visual patches",
            score=min(1.0, initial_score),
            agent_id=self.agent_id,
            path_history=[context]
        )
        self.current_hypotheses.append(new_h)
        
        if self.bus:
            logger.debug("agent_publish_hypothesis", agent_id=self.agent_id, h_id=h_id)
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
