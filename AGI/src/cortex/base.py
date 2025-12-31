from abc import ABC, abstractmethod
from typing import List
from AGI.src.bridge.schemas import VisualSegment

class VisualCortexBase(ABC):
    @abstractmethod
    def process(self, image_path: str) -> List[VisualSegment]:
        """Process an image file and return segmented embeddings."""
        pass
