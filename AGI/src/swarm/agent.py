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
        
        # Step 3: Cross-validation listener
        if self.bus:
            self.bus.subscribe("hypotheses", self.cross_validate)
        
    def perceive(self, tokens: List[AgentToken]):
        """
        Ingest tokens into the agent's memory.
        """
        logger.debug("agent_perceive", agent_id=self.agent_id, num_tokens=len(tokens))
        self.memory.extend(tokens)
        
    async def generate_candidate(self, context: str) -> Hypothesis:
        """
        Step 1: Candidate Generation.
        Propose a specific description segment based on the agent's local visual patches.
        """
        h_id = f"hyp_{uuid.uuid4().hex[:8]}"
        
        # Simulate 'seeing' something specific based on patch density or index
        # In a real system, this would be a CLIP-text matching or LLM call.
        content = f"Agent {self.agent_id[:4]} detects a high-contrast object at patch-sector {context}"
        if len(self.memory) > 10:
             content = f"Agent {self.agent_id[:4]} perceives a structural boundary (cube-like) in its sector."

        novelty_bonus = self.curiosity.score_hypothesis([context])
        initial_score = 0.5 + (0.1 * novelty_bonus)
        
        new_h = Hypothesis(
            hypothesis_id=h_id,
            content=content,
            score=initial_score,
            agent_id=self.agent_id,
            path_history=[context]
        )
        return new_h

    async def self_verify(self, hypothesis: Hypothesis):
        """
        Step 2: Self-Verification.
        The agent internalizes its own proposal and 'checks' it against its raw memory features.
        """
        # Simulated verification: if the content is specific, it's 'harder' to verify
        verification_boost = 0.1 if "structural" in hypothesis.content else 0.05
        hypothesis.score = min(1.0, hypothesis.score + verification_boost)
        hypothesis.evidence.append("Self-verified via patch feature consistency.")
        logger.debug("agent_self_verify", agent_id=self.agent_id, h_id=hypothesis.hypothesis_id, new_score=hypothesis.score)

    async def cross_validate(self, peer_hypothesis: Hypothesis):
        """
        Step 3: Cross-Validation.
        Listener callback. If another agent proposes something that matches this agent's memory, boost it.
        """
        if peer_hypothesis.agent_id == self.agent_id:
            return

        # Simple overlap logic: if both agents are looking at similar sectors or themes
        if "object" in peer_hypothesis.content and len(self.memory) > 5:
            peer_hypothesis.score = min(1.0, peer_hypothesis.score + 0.05)
            peer_hypothesis.evidence.append(f"Cross-validated by Agent {self.agent_id[:4]}")
            logger.debug("agent_cross_validate", observer=self.agent_id, target_h=peer_hypothesis.hypothesis_id)

    async def run_reasoning_step(self, context: str) -> Hypothesis:
        """
        Executes the internal 3-step sequence (Generate -> Verify -> Publish).
        """
        # 1. Generate
        candidate = await self.generate_candidate(context)
        
        # 2. Self-Verify
        await self.self_verify(candidate)
        
        # 3. Publish (for Step 4 cross-validation by others)
        if self.bus:
            await self.bus.publish("hypotheses", candidate)
            
        return candidate

    def verify_hypothesis(self, hypothesis: Hypothesis) -> float:
        """
        Self-verify a hypothesis. Return a new score.
        'Omnidirectional' aspect: can look at 'future' predictions (mocked here).
        """
        # Mocking omnidirectional lookahead/lookback
        influence = 0.1 if "future" in hypothesis.content else -0.1
        new_score = min(1.0, max(0.0, hypothesis.score + influence))
        return new_score
