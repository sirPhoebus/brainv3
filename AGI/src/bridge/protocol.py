import time
from typing import List
from AGI.src.bridge.schemas import VisualSegment, AgentToken

class Bridge:
    """
    Translates VisualSegments into AgentTokens.
    """
    
    @staticmethod
    def translate_segment(segment: VisualSegment) -> AgentToken:
        """
        Convert a single VisualSegment to an AgentToken.
        """
        return AgentToken(
            token_id=f"token_{segment.segment_id}",
            vector=segment.embedding,
            context_ref=segment.segment_id,
            timestamp=time.time(),
            priority=1.0, # Default priority
            metadata=segment.metadata
        )
    
    @staticmethod
    def translate_batch(segments: List[VisualSegment]) -> List[AgentToken]:
        """
        Convert a batch of VisualSegments to AgentTokens.
        """
        return [Bridge.translate_segment(s) for s in segments]
