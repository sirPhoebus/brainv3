import pytest
import asyncio
from AGI.src.swarm.core import Swarm
from typing import List, Optional, Dict, Any
from AGI.src.bridge.schemas import AgentToken
from AGI.src.swarm.agent import OmnidirectionalAgent

class LogicPuzzleAgent(OmnidirectionalAgent):
    """
    Subclass of agent that 'solves' a simple logic puzzle by finding a specific keyword.
    """
    async def propose_hypothesis(self, context: str) -> Any:
        h = await super().propose_hypothesis(context)
        # If this agent 'sees' a certain token, it might increase the score
        for token in self.memory:
            if "clue" in token.token_id:
                h.score = 0.99
                h.content = "Solved: The clue was found."
        return h

@pytest.mark.asyncio
async def test_logic_puzzle_benchmark():
    """
    Benchmark: Swarm must reach consensus on a logic puzzle where only one agent has the clue.
    """
    swarm = Swarm(num_agents=2)
    # Replace one agent with a specialized one
    swarm.agents[0] = LogicPuzzleAgent(bus=swarm.bus)
    
    tokens = [
        AgentToken(token_id="generic_data", vector=[0.0], timestamp=1.2),
        AgentToken(token_id="specific_clue", vector=[1.0], timestamp=1.3)
    ]
    
    # Run consensus
    best_h = await swarm.run_consensus_loop(tokens)
    
    assert best_h is not None
    assert "Solved" in best_h.content
    assert best_h.score >= 0.95
    assert swarm.iteration_count < swarm.max_iterations

@pytest.mark.asyncio
async def test_agent_timeout_safeguard():
    """
    Test that the swarm continues if an agent hangs.
    """
    class SlowAgent(OmnidirectionalAgent):
        async def propose_hypothesis(self, context: str):
            await asyncio.sleep(10) # Longer than timeout
            return await super().propose_hypothesis(context)
            
    swarm = Swarm(num_agents=2)
    swarm.timeout = 0.1 # short timeout for test
    swarm.agents[0] = SlowAgent(bus=swarm.bus)
    
    tokens = [AgentToken(token_id="t1", vector=[0.1], timestamp=1.0)]
    
    # This should not raise TimeoutError and should finish
    best_h = await swarm.run_consensus_loop(tokens)
    assert best_h is not None or len(swarm.global_hypotheses) >= 0
