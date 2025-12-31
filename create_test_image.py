from PIL import Image, ImageDraw

# Create a simple image with a blue circle on white background
img = Image.new('RGB', (512, 512), color='white')
draw = ImageDraw.Draw(img)
draw.ellipse([100, 100, 400, 400], fill='blue', outline='black', width=3)

img.save(r".\AGI\examples\blue_circle.png")
print("Created blue_circle.png - a simple blue circle on white background")
