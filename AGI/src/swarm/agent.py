import random
import uuid
import structlog
import torch
import numpy as np
from typing import List, Optional, Dict, Any, Set
from transformers import CLIPProcessor, CLIPModel
from AGI.src.swarm.schemas import Hypothesis, AgentAction
from AGI.src.bridge.schemas import AgentToken
from AGI.src.curiosity.scorer import CuriosityScorer
from AGI.src.swarm.predictor import ARCPredictor
from AGI.src.config_loader import DEFAULT_CONFIG

logger = structlog.get_logger()

class OmnidirectionalAgent:
    """
    An agent capable of reasoning across past and future states.
    """
    
    def __init__(self, bus: Any = None, agent_id: str = None, clip_model=None, clip_processor=None, task_data: Dict = None):
        self.config = DEFAULT_CONFIG.get("curiosity", {})
        self.agent_id = agent_id or str(uuid.uuid4())
        self.bus = bus
        self.task_data = task_data
        self.memory: List[AgentToken] = []
        self.active_hypotheses: Dict[str, Hypothesis] = {}
        self.seen_descriptions: Set[str] = set() 
        self.iteration = 0
        self.max_per_iter = 4
        self.curiosity = CuriosityScorer()

        # Rule Memory Integration
        from AGI.src.swarm.memory import RuleMemory
        self.rule_memory: Optional[RuleMemory] = None
        
        # Shared or new CLIP handles
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        
        # ARC-Specific Transformation Rule Bank
        self.prompt_bank = [
            "identity: output grid is identical to input grid",
            "reflection: mirror the top half of the input to the bottom output",
            "reflection: mirror the left half of the input to the right output",
            "color_fill: replace all 0-cells with the most frequent non-0 color",
            "translation: shift all colored objects 3 cells to the right",
            "pattern_continuation: continue the horizontal line until the edge",
            "object_detection: detect the largest cluster and surround with a border",
            "scaling: double the size of the input pattern in the output",
            "color_swap: change all colors of type X to type Y",
            "symmetry_completion: complete the partial symmetry around center",
            "gravity: move all objects to the bottom of the grid",
            "occlusion: hide objects that are behind the main central pattern"
        ]
        
        # New: task mode
        self.mode = "arc_reasoning" # Default to ARC for now
        
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
        Step 1: Data-Driven Candidate Generation.
        Propose candidates from a prompt bank and score them via CLIP similarity.
        """
        if not self.clip_model or not self.clip_processor:
            # Fallback if CLIP not loaded (lazy load or mock-ish)
             print("Warning: CLIP component missing in agent, using generic scoring.")
             return await self._generate_fallback_candidates(context)

        selected_prompts = []
        
        if self.rule_memory:
            # 50% from high-weight memory rules (top 10)
            high_weight = self.rule_memory.get_weighted_rules(top_n=10)
            if high_weight:
                num_high = self.max_per_iter // 2
                sampled_high = random.choices([r["text"] for r in high_weight], k=num_high)
                selected_prompts.extend(sampled_high)
            
            # 20% rehearsal from low-weight memory rules
            rehearsal = self.rule_memory.get_rehearsal_candidates(n=5)
            if rehearsal:
                num_rehearsal = max(1, self.max_per_iter // 5)
                sampled_rehearsal = random.choices([r["text"] for r in rehearsal], k=num_rehearsal)
                selected_prompts.extend(sampled_rehearsal)

        # Fill/Add from standard bank (30% or whatever is left)
        needed = self.max_per_iter - len(selected_prompts)
        if needed > 0:
            bank_samples = random.sample(self.prompt_bank, k=min(needed, len(self.prompt_bank)))
            selected_prompts.extend(bank_samples)

        # Final top-up if still under max
        if len(selected_prompts) < self.max_per_iter:
             extra = random.choices(self.prompt_bank, k=self.max_per_iter - len(selected_prompts))
             selected_prompts.extend(extra)
        
        # Dedupe while preserving order
        selected_prompts = list(dict.fromkeys(selected_prompts))[:self.max_per_iter]
        
        new_candidates = []

        # Gather evidence vectors from memory
        if not self.memory:
            return []
            
        # Sample patches for evidence
        num_samples = min(12, len(self.memory))
        evidence_tokens = random.sample(self.memory, num_samples)
        
        # Spatial weighting: Give higher weight to central patches for object identification
        # Metadata contains 'position_normalized': {'x': norm_x, 'y': norm_y}
        weights = []
        for t in evidence_tokens:
            pos = t.metadata.get('position_normalized', {'x': 0.5, 'y': 0.5})
            # Simple Gaussian-like weight, higher for center (0.5, 0.5)
            # Center is usually where the subject is in close-up photos
            dist_sq = (pos['x'] - 0.5)**2 + (pos['y'] - 0.5)**2
            weight = max(0.1, 1.0 - (dist_sq ** 0.5) * 1.5) 
            weights.append(weight)
            
        evidence_embeddings = torch.tensor([t.vector for t in evidence_tokens]).to(self.device).to(torch.float32)
        weights_tensor = torch.tensor(weights).to(self.device).to(torch.float32)
        
        for prompt_text in selected_prompts:
            # Curiosity boost
            curiosity_bonus = 0.2 if prompt_text not in self.seen_descriptions else 0.0
            self.seen_descriptions.add(prompt_text)
            
            # Compute CLIP similarity
            text_inputs = self.clip_processor(text=[prompt_text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_emb = self.clip_model.get_text_features(**text_inputs).to(torch.float32)
            
            # Normalize for cosine similarity
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            norm_evidence = evidence_embeddings / evidence_embeddings.norm(dim=-1, keepdim=True)
            
            # Similarities between text and each patch [batch=1, N_patches]
            similarities = torch.mm(text_emb, norm_evidence.T)[0]
            
            # Apply spatial weights
            weighted_sim = (similarities * weights_tensor).sum() / weights_tensor.sum()
            # Mapping clip score to 0-1 range
            # Memory boost: if rule was successful before, give it a head start
            is_prior = False
            if self.rule_memory:
                 is_prior = any(r["text"] == prompt_text for r in self.rule_memory.rules)
            
            memory_boost = 0.3 if is_prior else 0.0
            total_score = 0.4 + weighted_sim.item() * 0.4 + curiosity_bonus + memory_boost + random.uniform(-0.05, 0.05)
            
            h_id = f"hyp_{uuid.uuid4().hex[:12]}"
            hyp = Hypothesis(
                hypothesis_id=h_id,
                agent_id=self.agent_id,
                content=prompt_text,
                score=min(1.0, total_score),
                evidence=[t.token_id for t in evidence_tokens],
                iteration=self.iteration,
                metadata={"clip_raw_score": weighted_sim.item(), "context": context}
            )
            new_candidates.append(hyp)
            self.active_hypotheses[h_id] = hyp
            
        return new_candidates

    async def _generate_fallback_candidates(self, context: str) -> List[Hypothesis]:
        # Legacy placeholder logic if CLIP not injected
        return []

    async def self_verify(self, hypotheses: List[Hypothesis]):
        """
        Step 2: Self-Verification.
        Boost rules that correctly transform known demonstration pairs.
        """
        for hyp in hypotheses:
            # Basic grounded boost
            if len(hyp.evidence) >= 8:
                hyp.score = min(1.0, hyp.score + 0.1)
                hyp.evidence.append(f"Grounded: Supported by {len(hyp.evidence)} visual patches.")

            # Empirical boost if task_data is present
            if self.task_data and "input" in self.task_data and "output" in self.task_data:
                try:
                    predicted = ARCPredictor.apply_rule(hyp.content, self.task_data["input"])
                    if predicted == self.task_data["output"]:
                        hyp.score = min(1.0, hyp.score + 0.5) # Massive boost for correctness
                        hyp.evidence.append("Empirical Match: Rule correctly transforms input to output.")
                    else:
                        # Slight penalty for incorrect execution if we are sure of the mapping
                        # but be careful with DSL limitations
                        pass
                except Exception as e:
                    # If rule can't be executed yet, no boost or penalty
                    pass

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
