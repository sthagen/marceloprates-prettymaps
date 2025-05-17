#!/bin/bash

# Exit on error
set -e

NOTEBOOKS_DIR="notebooks"
README_FILE="README.md"
EXAMPLES_NOTEBOOK="$NOTEBOOKS_DIR/examples.ipynb"
PICTURES_DIR="pictures/README"

# Check if examples.ipynb exists
if [ ! -f "$EXAMPLES_NOTEBOOK" ]; then
  echo "examples.ipynb not found in $NOTEBOOKS_DIR. Exiting."
  exit 1
fi

# Create pictures/README directory if it doesn't exist
mkdir -p "$PICTURES_DIR"

# Export examples.ipynb to markdown, putting images in pictures/README
jupyter nbconvert --to markdown "$EXAMPLES_NOTEBOOK" --output temp_readme --output-dir "$PICTURES_DIR"

# Move the markdown file to the root as README.md
mv "$PICTURES_DIR/temp_readme.md" "$README_FILE"

# Move all images to pictures/README (they are already there, but this ensures they stay organized)
# (No-op if already in place)

# Update image links in README.md to point to pictures/README/
sed -i 's|(temp_readme_files/|(pictures/README/temp_readme_files/|g' "$README_FILE"

# Optionally, clean up any old temp_readme_files in root
# rm -rf temp_readme_files

echo "README.md has been replaced with the markdown export of $EXAMPLES_NOTEBOOK, and images are in $PICTURES_DIR." 