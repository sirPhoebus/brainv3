import asyncio
from typing import List, Dict
import structlog
from AGI.src.swarm.agent import OmnidirectionalAgent
from AGI.src.swarm.schemas import Hypothesis
from AGI.src.bridge.schemas import AgentToken
from AGI.src.swarm.comms import MessageBus
from AGI.src.swarm.verifier import SwarmVerifier

logger = structlog.get_logger()

class Swarm:
    """
    Orchestrates a collection of agents to reach a consensus.
    """
    
    def __init__(self, num_agents: int = 5):
        self.bus = MessageBus()
        self.agents = [OmnidirectionalAgent(bus=self.bus) for _ in range(num_agents)]
        self.global_hypotheses: List[Hypothesis] = []
        self.iteration_count = 0
        self.max_iterations = 20
        
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
            
            # 1. Propose Phase (Parallel)
            tasks = [agent.propose_hypothesis(context=f"step_{i}") for agent in self.agents]
            await asyncio.gather(*tasks)
                
            # 2. Cross-validation / Voting (Consensus)
            # Hypotheses are already being added to global_hypotheses via the bus callback
            
            # 3. Verification Phase
            self.global_hypotheses = SwarmVerifier.verify_consistency(self.global_hypotheses)
            self.global_hypotheses = SwarmVerifier.prune_conflicts(self.global_hypotheses)
            
            self._prune_hypotheses()
            
            # Check for early stopping (e.g., high consensus on top hypothesis)
            if self._check_convergence():
                logger.info("consensus_reached", step=i)
                break
                
        return self._get_best_hypothesis()

    def _prune_hypotheses(self):
        """
        Remove low-scoring hypotheses.
        """
        # Simple threshold pruning
        self.global_hypotheses = [h for h in self.global_hypotheses if h.score > 0.3]
        # Keep top N
        self.global_hypotheses.sort(key=lambda x: x.score, reverse=True)
        self.global_hypotheses = self.global_hypotheses[:50]

    def _check_convergence(self) -> bool:
        """
        Check if the swarm has converged on a solution.
        """
        if not self.global_hypotheses:
            return False
        return self.global_hypotheses[0].score > 0.95

    def _get_best_hypothesis(self) -> Hypothesis:
        if not self.global_hypotheses:
            return None
        return self.global_hypotheses[0]
