import pytest
from AGI.src.bridge.schemas import VisualSegment
from AGI.src.bridge.protocol import Bridge

def test_translate_segment():
    segment = VisualSegment(
        segment_id="test_1",
        embedding=[0.1, 0.2, 0.3],
        metadata={"key": "val"}
    )
    token = Bridge.translate_segment(segment)
    
    assert token.token_id == "token_test_1"
    assert token.vector == segment.embedding
    assert token.context_ref == "test_1"
    assert token.timestamp > 0

def test_translate_batch():
    segments = [
        VisualSegment(segment_id="s1", embedding=[0.1]),
        VisualSegment(segment_id="s2", embedding=[0.2])
    ]
    tokens = Bridge.translate_batch(segments)
    assert len(tokens) == 2
    assert tokens[0].token_id == "token_s1"
    assert tokens[1].token_id == "token_s2"
