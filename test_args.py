import argparse
import sys

print("Command line arguments received:")
print(f"sys.argv = {sys.argv}")

parser = argparse.ArgumentParser()
parser.add_argument('--image-path', type=str, default="default_path.png")
args = parser.parse_args()

print(f"\nParsed --image-path: {args.image_path}")
