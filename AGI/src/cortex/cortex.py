import uuid
from typing import List, Any
from AGI.src.cortex.base import VisualCortex
from AGI.src.bridge.schemas import VisualSegment

class MockCortex(VisualCortex):
    """
    A mock implementation of the Visual Cortex for testing.
    """
    
    def process_input(self, data: Any) -> List[VisualSegment]:
        """
        Generates dummy segments from any input.
        """
        segments = []
        # Simulate finding 3 segments
        for i in range(3):
            segments.append(VisualSegment(
                segment_id=f"seg_{uuid.uuid4().hex[:6]}",
                embedding=[0.1 * i] * 512, # Dummy 512d embedding
                metadata={"type": "mock_segment", "index": i}
            ))
        return segments

    def get_status(self) -> str:
        return "MockCortex: Active"
