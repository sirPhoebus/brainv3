from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any, Dict
import numpy as np

class VisualSegment(BaseModel):
    """
    Representation of an input chunk from the Visual Cortex.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    segment_id: str
    embedding: List[float] = Field(..., description="Vector representation of the segment")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata like coordinates, scale, or source type")
    content_summary: Optional[str] = None

class AgentToken(BaseModel):
    """
    Representation of a token consumable by the Swarm Agents.
    """
    token_id: str
    vector: List[float]
    context_ref: Optional[str] = Field(None, description="Reference back to the original VisualSegment or source")
    timestamp: float
    priority: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
