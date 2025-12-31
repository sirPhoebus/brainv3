import pytest
import asyncio
from AGI.src.swarm.core import Swarm
from AGI.src.bridge.schemas import AgentToken

@pytest.mark.asyncio
async def test_swarm_consensus():
    swarm = Swarm(num_agents=3)
    tokens = [
        AgentToken(token_id="t1", vector=[0.1], timestamp=1.0),
        AgentToken(token_id="t2", vector=[0.2], timestamp=1.1)
    ]
    
    best_h = await swarm.run_consensus_loop(tokens)
    
    assert best_h is not None
    assert best_h.score >= 0.5
    assert len(swarm.agents) == 3

@pytest.mark.asyncio
async def test_swarm_early_stopping():
    swarm = Swarm(num_agents=2)
    # Mock convergence
    tokens = [AgentToken(token_id="t1", vector=[0.1], timestamp=1.0)]
    
    # Manually inject a high-scoring hypothesis to trigger convergence
    from AGI.src.swarm.schemas import Hypothesis
    swarm.global_hypotheses.append(Hypothesis(
        hypothesis_id="winner",
        content="Winning",
        score=0.96,
        agent_id="agent_1"
    ))
    
    best_h = await swarm.run_consensus_loop(tokens)
    assert swarm.iteration_count < swarm.max_iterations - 1
