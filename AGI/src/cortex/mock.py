import uuid
from typing import List, Any
from AGI.src.cortex.base import VisualCortexBase
from AGI.src.bridge.schemas import VisualSegment

class MockCortex(VisualCortexBase):
    """
    Mock implementation for testing and development.
    """
    def process(self, image_path: Any) -> List[VisualSegment]:
        segments = []
        for i in range(3):
            segments.append(VisualSegment(
                segment_id=f"mock_{uuid.uuid4().hex[:6]}",
                embedding=[0.1 * i] * 512,
                metadata={"type": "mock_segment", "index": i}
            ))
        return segments

    def get_status(self) -> str:
        return "MockCortex: Active"
