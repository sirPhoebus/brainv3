import subprocess

result = subprocess.run(
    ["python", "-m", "AGI.src.main", "--image-path", r".\AGI\examples\blue_circle.png"],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

# Extract summary
lines = result.stdout.split('\n') + result.stderr.split('\n')

# Find the summary section
for i, line in enumerate(lines):
    if "AGI SYSTEM - ANALYSIS COMPLETE" in line:
        # Print 15 lines from summary
        for j in range(i, min(i+15, len(lines))):
            print(lines[j])
        break
