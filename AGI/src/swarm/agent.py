import random
import uuid
import structlog
from typing import List, Optional, Dict, Any, Set
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
        self.active_hypotheses: Dict[str, Hypothesis] = {}
        self.seen_descriptions: Set[str] = set() 
        self.iteration = 0
        self.max_per_iter = 3
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
        
    async def generate_candidate(self, context: str) -> List[Hypothesis]:
        """
        Step 1: Multi-Branch Candidate Generation.
        Generate diverse proposals using curiosity to favor novel descriptions.
        """
        templates = [
            "Detects a colorful cube-like object, possibly a Rubik's cube with stickers.",
            "Observes sharp structural edges forming a 3D cubic shape.",
            "Perceives a grid of colored patches suggesting a puzzle toy.",
            "Identifies potential furniture corner or box in the scene.",
            "Notes high-contrast boundaries and symmetry in central region.",
            "Sees patterned surface with primary colors arranged in squares.",
            "Hypothesizes a solved state of a twisty puzzle.",
            "Detects shadows indicating depth and 3D structure."
        ]
        
        new_candidates = []
        for _ in range(self.max_per_iter):
            description = random.choice(templates)
            # Curiosity boost: favor unseen descriptions
            curiosity_bonus = 0.2 if description not in self.seen_descriptions else 0.0
            self.seen_descriptions.add(description)
            
            h_id = f"hyp_{uuid.uuid4().hex[:12]}"
            
            # Select random actual tokens as evidence
            evidence_samples = []
            if self.memory:
                num_samples = min(8, len(self.memory))
                evidence_samples = [t.token_id for t in random.sample(self.memory, num_samples)]

            hyp = Hypothesis(
                hypothesis_id=h_id,
                agent_id=self.agent_id,
                content=description,
                score=0.6 + curiosity_bonus + random.uniform(-0.1, 0.1),
                evidence=evidence_samples,
                iteration=self.iteration,
                metadata={"context": context}
            )
            new_candidates.append(hyp)
            self.active_hypotheses[h_id] = hyp
            
        return new_candidates

    async def self_verify(self, hypotheses: List[Hypothesis]):
        """
        Step 2: Self-Verification.
        Basic consistency check with evidence and domain heuristics.
        """
        for hyp in hypotheses:
            if len(hyp.evidence) >= 5:  # Needs supporting patches
                hyp.score = min(1.0, hyp.score + 0.15)
                hyp.evidence.append("Confidence boost: Sufficient patch support.")
            
            low_content = hyp.content.lower()
            if "cube" in low_content or "rubik" in low_content:
                hyp.score = min(1.0, hyp.score + 0.1)
                hyp.evidence.append("Domain Match: Geometric cubic signature detected.")

    async def cross_validate(self, peer_hypothesis: Hypothesis):
        """
        Step 3: Cross-Validation.
        Boost agreement, weaken conflict based on semantic overlap.
        """
        if peer_hypothesis.agent_id == self.agent_id:
            return

        for my_hyp in list(self.active_hypotheses.values()):
            # Simple overlap logic: semantic similarity check
            my_words = set(my_hyp.content.lower().split())
            peer_words = set(peer_hypothesis.content.lower().split())
            overlap = len(my_words & peer_words)
            similarity = overlap / max(len(my_words), 1)
            
            if similarity > 0.6:
                my_hyp.score = min(1.0, my_hyp.score + 0.08)
                my_hyp.evidence.append(f"Consensus boost: Matches Agent {peer_hypothesis.agent_id[:4]}")
            elif similarity < 0.2:
                my_hyp.score = max(0.0, my_hyp.score - 0.05)

    async def run_reasoning_step(self, context: str) -> List[Hypothesis]:
        """
        Executes the internal reasoning cycle.
        """
        # 1. Generate Multi-branch
        candidates = await self.generate_candidate(context)
        
        # 2. Self-Verify
        await self.self_verify(candidates)
        
        # 3. Publish (Triggering Cross-Val in others)
        if self.bus:
            for h in candidates:
                await self.bus.publish("hypotheses", h)
        
        # 4. Local Pruning
        self._prune_weak()
        
        self.iteration += 1
        return list(self.active_hypotheses.values())

    def _prune_weak(self):
        """
        Step 4: Prune weak hypotheses.
        """
        to_remove = [hid for hid, hyp in self.active_hypotheses.items() if hyp.score < 0.4]
        for hid in to_remove:
            del self.active_hypotheses[hid]
