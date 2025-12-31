from abc import ABC, abstractmethod
from typing import List, Any
from AGI.src.bridge.schemas import VisualSegment

class VisualCortex(ABC):
    """
    Abstract Base Class for input processing.
    """
    
    @abstractmethod
    def process_input(self, data: Any) -> List[VisualSegment]:
        """
        Process raw input data into a list of VisualSegment objects.
        """
        pass

    @abstractmethod
    def get_status(self) -> str:
        """
        Return the current status of the cortex.
        """
        pass
