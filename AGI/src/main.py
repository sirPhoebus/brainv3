import asyncio
import structlog
from AGI.src.bridge.protocol import Bridge
from AGI.src.swarm.core import Swarm
from AGI.src.hitl.interface import HITLInterface
from AGI.src.cortex import VisualCortex

# Set up logging
structlog.configure()
logger = structlog.get_logger()

async def main():
    logger.info("starting_agi_system")
    
    # 1. Initialize Components
    from AGI.src.cortex import VisualCortex
    cortex = VisualCortex() 
    swarm = Swarm(num_agents=3, 
                  clip_model=getattr(cortex, 'model', None), 
                  clip_processor=getattr(cortex, 'processor', None))
    
    # 2. Process Input
    import os
    image_path = "AGI/examples/sample_image.png"
    if not os.path.exists(image_path):
        logger.warning("sample_image_not_found_using_mock", path=image_path)
        from AGI.src.cortex.mock import MockCortex
        cortex = MockCortex()
        image_path = "mock_data"
        
    logger.info("processing_input", path=image_path)
    segments = cortex.process(image_path)

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
