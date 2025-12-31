from PIL import Image
import os

image_path = r"C:\Users\pscho\Downloads\grid2x2.png"

if os.path.exists(image_path):
    img = Image.open(image_path)
    file_size = os.path.getsize(image_path)
    print(f"Image Path: {image_path}")
    print(f"Dimensions: {img.width}x{img.height}")
    print(f"Format: {img.format}")
    print(f"Mode: {img.mode}")
    print(f"File Size: {round(file_size / 1024, 2)} KB")
    print(f"\nImage exists and is readable!")
else:
    print(f"Image not found at: {image_path}")
