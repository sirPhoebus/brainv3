import subprocess
import re

def test_image(path):
    result = subprocess.run(
        ["python", "-m", "AGI.src.main", "--image-path", path],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=30
    )
    
    output = result.stdout + result.stderr
    
    # Extract ANALYSIS RESULT
    match = re.search(r'🧠 ANALYSIS RESULT:\s+(.+)', output)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for reasoning_complete
    match = re.search(r'reasoning_complete.*?content="([^"]+)"', output)
    if match:
        return match.group(1).strip()
    
    return "No result found"

print("Testing different images with expanded prompt bank...")
print("="*70)

images = {
    "Blue Circle": r".\AGI\examples\blue_circle.png",
    "Grid 2x2": r".\AGI\examples\grid2x2.png",
}

for name, path in images.items():
    print(f"\n{name}: {path}")
    try:
        result = test_image(path)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error: {str(e)}")

print("\n" + "="*70)
