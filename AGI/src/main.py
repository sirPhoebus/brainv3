import asyncio
import structlog
from AGI.src.cortex.cortex import MockCortex
from AGI.src.bridge.protocol import Bridge
from AGI.src.swarm.core import Swarm
from AGI.src.hitl.interface import HITLInterface

# Set up logging
structlog.configure()
logger = structlog.get_logger()

async def main():
    logger.info("starting_agi_system")
    
    # 1. Initialize Components
    # CLIPVisualCortex is exported as VisualCortex by default in AGI.src.cortex
    cortex = MockCortex() # Defaulting to Mock for first test, then switch to real
    swarm = Swarm(num_agents=3)
    
    # 2. Process Input
    # Use real image path if it exists, otherwise fallback to mock data via raw_data
    import os
    image_path = "AGI/examples/sample_image.png"
    if os.path.exists(image_path):
        logger.info("using_real_image", path=image_path)
        # Switch to real cortex for the demo
        from AGI.src.cortex import VisualCortex
        cortex = VisualCortex()
        segments = cortex.process_input(image_path)
    else:
        logger.warning("sample_image_not_found_using_mock", path=image_path)
        raw_data = "Simulated visual input"
        segments = cortex.process_input(raw_data)
    logger.info("input_processed", num_segments=len(segments))
    
    # 3. Translate to tokens via Bridge
    tokens = Bridge.translate_batch(segments)
    logger.info("tokens_generated", num_tokens=len(tokens))
    
    # 4. Run Swarm reasoning
    logger.info("running_swarm_reasoning")
    best_hypothesis = await swarm.run_consensus_loop(tokens)
    
    # 5. HITL Review
    hitl = HITLInterface()
    if swarm.global_hypotheses:
        reviewed = await hitl.review_hypotheses(swarm.global_hypotheses)
        best_hypothesis = reviewed[0] # Update best after review
    
    if best_hypothesis:
        logger.info("reasoning_complete", 
                    hypothesis_id=best_hypothesis.hypothesis_id,
                    score=best_hypothesis.score,
                    content=best_hypothesis.content)
    else:
        logger.error("no_hypothesis_found")

if __name__ == "__main__":
    asyncio.run(main())
