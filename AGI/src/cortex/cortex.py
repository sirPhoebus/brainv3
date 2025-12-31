import torch
from PIL import Image
import numpy as np
import uuid
from transformers import CLIPProcessor, CLIPModel
from typing import List, Any, Tuple

from AGI.src.cortex.base import VisualCortexBase
from AGI.src.bridge.schemas import VisualSegment
from AGI.src.config_loader import DEFAULT_CONFIG

class CLIPVisualCortex(VisualCortexBase):
    """
    Real Visual Cortex implementation using OpenAI's CLIP model.
    """
    def __init__(self, model_name: str = None):
        self.config = DEFAULT_CONFIG.get("cortex", {})
        m_name = model_name or self.config.get("model_name", "openai/clip-vit-base-patch32")
        print(f"Initializing CLIP model: {m_name}...")
        self.model = CLIPModel.from_pretrained(m_name)
        self.processor = CLIPProcessor.from_pretrained(m_name)
        self.patch_size = self.config.get("patch_size", 32)  # CLIP-ViT-B/32 uses 32x32 patches
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Cortex initialized on {self.device}")

    def _grid_patch_embeddings(self, image: Image.Image) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        """Simple grid-based 'segmentation': crop image into patches and embed each."""
        # CLIP expected size is 224x224
        resized = image.resize((224, 224))
        embeddings = []
        patch_coords = []
        
        patch_wh = self.patch_size
        
        # Process patches
        # NOTE: A real implementation might use the patched vision transformer features directly,
        # but here we follow the "segmentation" approach from phases.md for clarity.
        for y in range(0, 224, patch_wh):
            for x in range(0, 224, patch_wh):
                patch = resized.crop((x, y, x + patch_wh, y + patch_wh))
                patch_inputs = self.processor(images=patch, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    patch_emb = self.model.get_image_features(**patch_inputs)[0]
                embeddings.append(patch_emb.cpu().numpy())
                patch_coords.append((x / 224.0, y / 224.0))  # Normalized coords

        return embeddings, patch_coords

    def process_input(self, image_path: str) -> List[VisualSegment]:
        """
        Process an image file path and return patches as VisualSegments.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return []

        embeddings, coords = self._grid_patch_embeddings(image)

        segments = []
        for emb, (norm_x, norm_y) in zip(embeddings, coords):
            # Create a unique segment ID
            sid = f"clip_{uuid.uuid4().hex[:8]}"
            segments.append(
                VisualSegment(
                    segment_id=sid,
                    embedding=emb.tolist(),
                    metadata={
                        "type": "patch",
                        "position_normalized": {"x": norm_x, "y": norm_y},
                        "patch_size": self.patch_size,
                        "source": "clip_patch"
                    }
                )
            )
        return segments

    def get_status(self) -> str:
        return f"CLIPVisualCortex: Active on {self.device}"

class MockCortex(VisualCortexBase):
    """
    Keep MockCortex for fallback/testing.
    """
    def process_input(self, data: Any) -> List[VisualSegment]:
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
