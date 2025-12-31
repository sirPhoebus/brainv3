import asyncio
import argparse
import os
import structlog
from AGI.src.bridge.protocol import Bridge
from AGI.src.swarm.core import Swarm
from AGI.src.hitl.interface import HITLInterface
from AGI.src.cortex import VisualCortex

# Set up logging
structlog.configure()
logger = structlog.get_logger()

def parse_arguments():
    """Parse command-line arguments for image path."""
    parser = argparse.ArgumentParser(
        description="AGI System - Process images using swarm intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m AGI.src.main --image-path /path/to/your/image.png
  python -m AGI.src.main --image-path C:\\Users\\Username\\Pictures\\photo.jpg
  python -m AGI.src.main  # Uses default sample image
        """
    )
    parser.add_argument(
        '--image-path',
        type=str,
        default="AGI/examples/sample_image.png",
        help='Path to the input image file (default: AGI/examples/sample_image.png)'
    )
    return parser.parse_args()

async def main():
    # Parse command-line arguments
    args = parse_arguments()
    image_path = args.image_path
    
    logger.info("starting_agi_system", image_path=image_path)
    
    # 1. Initialize Components
    from AGI.src.cortex import VisualCortex
    cortex = VisualCortex() 
    swarm = Swarm(num_agents=3, 
                  clip_model=getattr(cortex, 'model', None), 
                  clip_processor=getattr(cortex, 'processor', None))
    
    # 2. Process Input
    # Validate image path exists
    if not os.path.exists(image_path):
        logger.warning("image_not_found_using_mock", path=image_path)
        from AGI.src.cortex.mock import MockCortex
        cortex = MockCortex()
        image_path = "mock_data"
    else:
        # Show image details to verify correct file is loaded
        from PIL import Image
        try:
            img = Image.open(image_path)
            file_size = os.path.getsize(image_path)
            logger.info("image_details", 
                       path=image_path,
                       dimensions=f"{img.width}x{img.height}",
                       format=img.format,
                       mode=img.mode,
                       file_size_kb=round(file_size / 1024, 2))
        except Exception as e:
            logger.warning("could_not_read_image_details", error=str(e))
        
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
        
        # Display clear summary
        print("\n" + "="*70)
        print(" AGI SYSTEM - ANALYSIS COMPLETE")
        print("="*70)
        print(f"\n📁 INPUT IMAGE: {image_path}")
        if os.path.exists(image_path):
            from PIL import Image
            img = Image.open(image_path)
            print(f"   Dimensions: {img.width}x{img.height}")
            print(f"   Format: {img.format}")
        print(f"\n🧠 ANALYSIS RESULT:")
        print(f"   {best_hypothesis.content}")
        print(f"\n📊 CONFIDENCE SCORE: {best_hypothesis.score:.2%}")
        print(f"   Hypothesis ID: {best_hypothesis.hypothesis_id}")
        print("\n" + "="*70 + "\n")
    else:
        logger.error("no_hypothesis_found")

if __name__ == "__main__":
    asyncio.run(main())
