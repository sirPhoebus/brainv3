import asyncio
import argparse
import structlog
from AGI.src.bridge.protocol import Bridge
from AGI.src.swarm.core import Swarm
from AGI.src.hitl.interface import HITLInterface
from AGI.src.cortex import VisualCortex

# Set up logging
structlog.configure()
logger = structlog.get_logger()

async def main(image_path: str = "AGI/examples/arc_tasks/task_user_composite.png"):
    logger.info("starting_agi_system")
    
    # 1. Initialize Components
    from AGI.src.cortex import VisualCortex
    cortex = VisualCortex() 
    # Example ARC Task Data (Mirroring)
    user_task = {
        "input": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
        "output": [[0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 6, 0, 0, 0, 0, 0], [0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0], [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0], [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]]
    }

    swarm = Swarm(num_agents=3, 
                  clip_model=getattr(cortex, 'model', None), 
                  clip_processor=getattr(cortex, 'processor', None),
                  task_data=user_task)
    
    # 2. Process Input
    import os
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
    # For ARC, we use the composite image to discover the rule
    best_hypothesis = await swarm.run_consensus_loop(tokens)
    
    # Example ARC Input Grid (the center object from task_user)
    # This would typically come from the task JSON
    sample_arc_input = [[0]*16 for _ in range(16)]
    sample_arc_input[7] = [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0]

    # 5. Apply Prediction
    if best_hypothesis:
        # If it's an ARC hypothesis, execute it
        prediction = await swarm.apply_prediction(best_hypothesis, sample_arc_input)
        logger.info("arc_prediction_generated", rule=best_hypothesis.content)
        
        # Visualize prediction
        from AGI.utils.arc_renderer import save_prediction
        save_prediction(prediction, "AGI/examples/arc_tasks/last_prediction.png")
    
    # 6. HITL Review
    hitl = HITLInterface()
    if swarm.global_hypotheses:
        reviewed = await hitl.review_hypotheses(swarm.global_hypotheses)
        best_hypothesis = reviewed[0] # Update best after review
        
        # 7. Distillation: Persist successful rule to memory
        if best_hypothesis.score > 0.8:
            logger.info("distilling_successful_rule", rule=best_hypothesis.content)
            swarm.rule_memory.persist_rule(best_hypothesis.content)
    
    if best_hypothesis:
        logger.info("reasoning_complete", 
                    hypothesis_id=best_hypothesis.hypothesis_id,
                    score=best_hypothesis.score,
                    content=best_hypothesis.content)
    else:
        logger.error("no_hypothesis_found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AGI System - ARC Task Solver with Visual Cortex and Swarm Reasoning"
    )
    parser.add_argument(
        "--image-path", 
        "-i", 
        type=str, 
        default="AGI/examples/arc_tasks/task_user_composite.png",
        help="Path to the input image file (default: AGI/examples/arc_tasks/task_user_composite.png)"
    )
    
    args = parser.parse_args()
    asyncio.run(main(image_path=args.image_path))

