import asyncio
from typing import List, Dict
import structlog
from AGI.src.swarm.agent import OmnidirectionalAgent
from AGI.src.swarm.schemas import Hypothesis
from AGI.src.bridge.schemas import AgentToken
from AGI.src.swarm.comms import MessageBus
from AGI.src.config_loader import DEFAULT_CONFIG
from AGI.src.swarm.predictor import ARCPredictor
from AGI.src.swarm.memory import RuleMemory
from AGI.src.swarm.verifier import SwarmVerifier
import torch
import uuid

logger = structlog.get_logger()

class Swarm:
    """
    Orchestrates a collection of agents to reach a consensus.
    """
    
    def __init__(self, num_agents: int = None, clip_model = None, clip_processor = None):
        self.config = DEFAULT_CONFIG.get("swarm", {})
        n_agents = num_agents or self.config.get("num_agents", 5)
        
        self.bus = MessageBus()
        self.rule_memory = RuleMemory()
        
        # Pull rules from memory to bias agents
        top_rules = self.rule_memory.get_top_rules()
        self.agents = [OmnidirectionalAgent(bus=self.bus, 
                                            clip_model=clip_model, 
                                            clip_processor=clip_processor) 
                       for _ in range(n_agents)]
        
        # Inject known rules into agent prompt banks
        for agent in self.agents:
            for rule in top_rules:
                if rule not in agent.prompt_bank:
                    agent.prompt_bank.append(rule)

        self.global_hypotheses: List[Hypothesis] = []
        self.iteration_count = 0
        self.max_iterations = self.config.get("max_iterations", 20)
        self.timeout = self.config.get("agent_timeout_seconds", 5.0)        
        # Subscribe to new hypotheses
        self.bus.subscribe("hypotheses", self._handle_new_hypothesis)

    async def _handle_new_hypothesis(self, hypothesis: Hypothesis):
        """
        Callback when any agent publishes a hypothesis.
        """
        self.global_hypotheses.append(hypothesis)
        logger.debug("received_hypothesis", hypothesis_id=hypothesis.hypothesis_id)
        
    async def run_consensus_loop(self, input_tokens: List[AgentToken]):
        """
        Main reasoning loop.
        """
        logger.info("starting_consensus_loop", num_agents=len(self.agents), num_tokens=len(input_tokens))
        
        # Initial perception
        for agent in self.agents:
            agent.perceive(input_tokens)
            
        for i in range(self.max_iterations):
            self.iteration_count = i
            logger.debug("iteration_step", step=i)
            
            # Step 1, 2, 3: Candidate Generation -> Self-Verify -> Publish (Cross-Val)
            # Agents perform internal reasoning and publish candidates to the bus.
            tasks = [asyncio.wait_for(agent.run_reasoning_step(context=f"sector_{i}"), timeout=self.timeout) 
                     for agent in self.agents]
            try:
                # This gathers and triggers cross-validation listeners on the bus simultaneously
                await asyncio.gather(*tasks)
            except asyncio.TimeoutError:
                logger.warning("agent_timeout_during_reasoning", step=i)
            except Exception as e:
                logger.error("agent_unhandled_error", error=str(e))
                
            # Step 4: Swarm-level Pruning and Strengthening
            # Strengthen based on global consensus markers and merge similar findings
            self.global_hypotheses = SwarmVerifier.verify_consistency(self.global_hypotheses)
            self.global_hypotheses = SwarmVerifier.merge_similar(self.global_hypotheses)
            self.global_hypotheses = SwarmVerifier.prune_conflicts(self.global_hypotheses)
            
            # Additional cleanup (top K)
            self._prune_hypotheses()
            
            # Step 5: Check for Early Consensus (Stop Iterating)
            if self._check_convergence():
                logger.info("consensus_reached", step=i, top_score=self.global_hypotheses[0].score)
                break
                
        # Final Refinement: Global alignment and Synthesis
        final_h = await self._synthesize_final_hypothesis(input_tokens)
        
        # ARC Specific: Prediction Execution
        # We assume the last task context or a global state provides the grid
        # For prototype, we'll try to apply the rule to a dummy input if needed
        # but the real value is show it in HITL
        return final_h

    async def apply_prediction(self, hypothesis: Hypothesis, input_grid: List[List[int]]) -> List[List[int]]:
        """
        Executes the rule in the hypothesis using the ARCPredictor.
        """
        logger.info("applying_transformation_rule", rule=hypothesis.content)
        predicted = ARCPredictor.apply_rule(hypothesis.content, input_grid)
        return predicted

    async def _synthesize_final_hypothesis(self, all_tokens: List[AgentToken]) -> Hypothesis:
        """
        Synthesizes a final rich hypothesis by merging top consensus results
        and calculating global CLIP alignment.
        """
        if not self.global_hypotheses:
            return None

        top_hyps = self.global_hypotheses[:3]
        primary = top_hyps[0]
        
        # Combine descriptions if they are unique enough
        additional_info = []
        seen_words = set(primary.content.lower().split())
        for h in top_hyps[1:]:
            h_words = set(h.content.lower().split())
            if len(h_words - seen_words) > 2: # Significant unique info
                additional_info.append(h.content)
                seen_words.update(h_words)
        
        final_content = primary.content
        if additional_info:
             # Very simple combination
             final_content = f"{primary.content}, including {additional_info[0]}"

        # Global alignment: CLIP similarity against MEAN of ALL patches
        clip_model = self.agents[0].clip_model
        clip_processor = self.agents[0].clip_processor
        device = self.agents[0].device

        if clip_model and clip_processor and all_tokens:
            all_vectors = torch.tensor([t.vector for t in all_tokens]).to(device).to(torch.float32)
            mean_emb = all_vectors.mean(dim=0, keepdim=True)
            mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
            
            text_inputs = clip_processor(text=[final_content], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text_emb = clip_model.get_text_features(**text_inputs).to(torch.float32)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            
            global_sim = torch.mm(text_emb, mean_emb.T).item()
            logger.info("global_alignment_check", score=global_sim, content=final_content)
        
        # Create synthesized result
        final = Hypothesis(
            hypothesis_id=f"final_{uuid.uuid4().hex[:8]}",
            content=final_content,
            score=primary.score,
            agent_id="swarm_sync",
            evidence=primary.evidence
        )
        # Add to global pool so it shows up in HITL
        self.global_hypotheses.insert(0, final)
        return final

    def _prune_hypotheses(self):
        """
        Remove low-scoring hypotheses based on config thresholds.
        """
        threshold = self.config.get("pruning_threshold", 0.3)
        max_keep = self.config.get("max_hypotheses_keep", 50)
        
        # Simple threshold pruning
        self.global_hypotheses = [h for h in self.global_hypotheses if h.score > threshold]
        # Keep top N
        self.global_hypotheses.sort(key=lambda x: x.score, reverse=True)
        self.global_hypotheses = self.global_hypotheses[:max_keep]

    def _check_convergence(self) -> bool:
        """
        Check if the swarm has converged.
        Stop when >=80% of top hypotheses share high similarity or iterations hit max.
        """
        if len(self.global_hypotheses) < 5:
            return False
            
        # Analyze top 10
        top_candidates = self.global_hypotheses[:10]
        consensus_count = 0
        primary_words = set(top_candidates[0].content.lower().split())
        
        for h in top_candidates:
            h_words = set(h.content.lower().split())
            overlap = len(primary_words & h_words)
            similarity = overlap / max(len(primary_words), 1)
            if similarity > 0.7:
                consensus_count += 1
                
        # Convergence if 80% of top group agrees
        convergence_ratio = consensus_count / len(top_candidates)
        threshold = self.config.get("convergence_threshold", 0.8)
        
        return convergence_ratio >= threshold or self.global_hypotheses[0].score > 0.98

    def _get_best_hypothesis(self) -> Hypothesis:
        if not self.global_hypotheses:
            return None
        return self.global_hypotheses[0]
