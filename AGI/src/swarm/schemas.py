from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Hypothesis(BaseModel):
    """
    Representation of a partial solution or reasoning path.
    """
    hypothesis_id: str
    content: str
    score: float = 0.0
    evidence: List[str] = Field(default_factory=list)
    path_history: List[str] = Field(default_factory=list)
    agent_id: str
    iteration: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentAction(BaseModel):
    """
    Action taken by an agent (e.g., query, propose).
    """
    agent_id: str
    action_type: str
    payload: Any
    timestamp: float
