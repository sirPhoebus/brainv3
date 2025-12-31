from PIL import Image
import numpy as np

image_path = r".\AGI\examples\grid2x2.png"

img = Image.open(image_path)
img_array = np.array(img)

print(f"Image: {image_path}")
print(f"Dimensions: {img.width}x{img.height}")
print(f"Mode: {img.mode}")
print(f"\nColor statistics:")
print(f"Unique colors: {len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))}")
print(f"\nFirst 100x100 pixel sample (top-left corner):")
sample = img_array[:100, :100]
unique_sample_colors = np.unique(sample.reshape(-1, sample.shape[2]), axis=0)
print(f"Unique colors in sample: {len(unique_sample_colors)}")
print(f"Sample colors (RGB):")
for i, color in enumerate(unique_sample_colors[:10]):
    print(f"  Color {i+1}: RGB{tuple(color)}")

# Save a smaller thumbnail to inspect
thumbnail = img.copy()
thumbnail.thumbnail((200, 200))
thumbnail.save("grid2x2_thumbnail.png")
print(f"\nThumbnail saved to: grid2x2_thumbnail.png")
