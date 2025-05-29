import os
import shutil
import sys

def organize_images(target_dir):
    # List all files in the target directory
    for filename in os.listdir(target_dir):
        # Skip directories and hidden files
        if os.path.isdir(os.path.join(target_dir, filename)) or filename.startswith('.'):
            continue
        # Extract organ name (before first '-')
        organ = filename.split('-')[0]
        organ_dir = os.path.join(target_dir, organ)
        # Create organ directory if it doesn't exist
        os.makedirs(organ_dir, exist_ok=True)
        # Source and destination paths
        src = os.path.join(target_dir, filename)
        dst = os.path.join(organ_dir, filename)
        # Move the file
        shutil.move(src, dst)
        print(f"Moved {filename} to {organ_dir}/")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python organize_dataset.py <directory>")
        sys.exit(1)
    target_dir = sys.argv[1]
    organize_images(target_dir) 