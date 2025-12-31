import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple

from AGI.src.cortex.base import VisualCortexBase
from AGI.src.bridge.schemas import VisualSegment

class CLIPVisualCortex(VisualCortexBase):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP model '{model_name}' on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.grid_size = 224  # CLIP input size
        self.patch_size = 32  # For ViT-B/32

    def _extract_patch_embeddings(self, image: Image.Image) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        """Extract embeddings from grid patches for pseudo-segmentation."""
        # Resize while preserving aspect for consistent patching
        image_resized = image.resize((self.grid_size, self.grid_size))
        
        embeddings = []
        coords = []
        
        with torch.no_grad():
            for y in range(0, self.grid_size, self.patch_size):
                for x in range(0, self.grid_size, self.patch_size):
                    # Crop the patch
                    patch = image_resized.crop((x, y, x + self.patch_size, y + self.patch_size))
                    # Process patch
                    patch_inputs = self.processor(images=patch, return_tensors="pt").to(self.device)
                    # Extract features
                    patch_emb = self.model.get_image_features(**patch_inputs)[0].cpu().numpy()
                    embeddings.append(patch_emb)
                    # Normalized coordinates for spatial awareness
                    coords.append((x / self.grid_size, y / self.grid_size))
        
        return embeddings, coords

    def process(self, image_path: str) -> List[VisualSegment]:
        """
        Process an image file and return segmented embeddings.
        """
        # Clear any cached computations to ensure fresh processing
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # Clear model's cached states if any
        self.model.eval()  # Ensure eval mode
        
        image = Image.open(image_path).convert("RGB")
        embeddings, coords = self._extract_patch_embeddings(image)
        
        # Debug: Show embedding fingerprint to verify different images produce different embeddings
        if embeddings:
            import hashlib
            first_emb_str = str(embeddings[0][:10])  # First 10 values of first embedding
            emb_hash = hashlib.md5(first_emb_str.encode()).hexdigest()[:8]
            print(f"[DEBUG] Embedding fingerprint: {emb_hash} (from {image_path})")
        
        segments = []
        import uuid
        for emb, (norm_x, norm_y) in zip(embeddings, coords):
            segments.append(
                VisualSegment(
                    segment_id=f"clip_{uuid.uuid4().hex[:8]}",
                    embedding=emb.flatten().tolist(),  # Flatten to list[float]
                    metadata={
                        "type": "patch",
                        "position_normalized": {"x": norm_x, "y": norm_y},
                        "confidence": 0.95,  # Can enhance later with attention scores
                        "source": "clip_vit_patch"
                    }
                )
            )
        return segments
