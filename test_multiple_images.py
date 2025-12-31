import subprocess
import sys

# Test with three different images
images = [
    r".\AGI\examples\sample_image.png",
    r".\AGI\examples\grid2x2.png", 
    r".\AGI\examples\blue_circle.png"
]

print("\n" + "="*80)
print("TESTING AGI SYSTEM WITH DIFFERENT IMAGES")
print("="*80 + "\n")

for img_path in images:
    print(f"\n{'='*80}")
    print(f"Testing: {img_path}")
    print("=" *80)
    
    # Run the command
    result = subprocess.run(
        ["python", "-m", "AGI.src.main", "--image-path", img_path],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # Extract the summary section from output
    output_lines = result.stdout.split('\n') + result.stderr.split('\n')
    
    # Find and print the summary
    in_summary = False
    for line in output_lines:
        if "AGI SYSTEM - ANALYSIS COMPLETE" in line:
            in_summary = True
        if in_summary:
            print(line)
            if line.strip().startswith("====") and in_summary and "COMPLETE" not in line:
                break
    
    if not in_summary:
        # Try to find the reasoning_complete line
        for line in output_lines:
            if "reasoning_complete" in line:
                print(f"Result: {line}")
                break

print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)
