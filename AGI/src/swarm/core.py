import asyncio
from typing import List, Dict
import structlog
from AGI.src.swarm.agent import OmnidirectionalAgent
from AGI.src.swarm.schemas import Hypothesis
from AGI.src.bridge.schemas import AgentToken
from AGI.src.swarm.comms import MessageBus
from AGI.src.swarm.verifier import SwarmVerifier
from AGI.src.config_loader import DEFAULT_CONFIG

logger = structlog.get_logger()

class Swarm:
    """
    Orchestrates a collection of agents to reach a consensus.
    """
    
    def __init__(self, num_agents: int = None):
        self.config = DEFAULT_CONFIG.get("swarm", {})
        n_agents = num_agents or self.config.get("num_agents", 5)
        
        self.bus = MessageBus()
        self.agents = [OmnidirectionalAgent(bus=self.bus) for _ in range(n_agents)]
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
                
        return self._get_best_hypothesis()

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
