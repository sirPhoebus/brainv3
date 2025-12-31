from AGI.src.cortex import VisualCortex
import sys

image1 = r".\AGI\examples\sample_image.png"  
image2 = r".\AGI\examples\grid2x2.png"

print("="*70)
print("Testing if different images produce different embeddings")
print("="*70)

cortex = VisualCortex()

print(f"\n1. Processing: {image1}")
segments1 = cortex.process(image1)
print(f"   Generated {len(segments1)} segments")
print(f"   First segment embedding (first 5 values): {segments1[0].embedding[:5]}")

print(f"\n2. Processing: {image2}")
segments2 = cortex.process(image2)
print(f"   Generated {len(segments2)} segments")
print(f"   First segment embedding (first 5 values): {segments2[0].embedding[:5]}")

# Compare
print(f"\n3. Comparison:")
are_same = segments1[0].embedding == segments2[0].embedding
print(f"   Embeddings are {'IDENTICAL' if are_same else 'DIFFERENT'}")

if not are_same:
    print("\nSUCCESS: Different images produce different embeddings!")
else:
    print("\nPROBLEM: Same embeddings for different images - caching issue!")
