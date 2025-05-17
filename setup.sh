#!/bin/bash

set -e

# Detect OS
OS="$(uname)"
echo "Detected OS: $OS"

# Change permissions for SRTM1 folder (Linux/macOS only)
if [ -d "./SRTM1" ]; then
  chmod -R 755 ./SRTM1
fi

if [[ "$OS" == "Linux" ]]; then
  # Linux (Debian/Ubuntu)
  echo "Updating package list and installing dependencies for Linux..."
  sudo apt-get update
  sudo apt-get install -y build-essential make gcc libgdal-dev gdal-bin
  sudo apt-get clean

elif [[ "$OS" == "Darwin" ]]; then
  # macOS
  echo "Installing dependencies for macOS (requires Homebrew)..."
  if ! command -v brew &>/dev/null; then
    echo "Homebrew not found. Please install Homebrew first: https://brew.sh/"
    exit 1
  fi
  brew update
  brew install gdal make gcc

elif [[ "$OS" =~ "MINGW" || "$OS" =~ "MSYS" || "$OS" =~ "CYGWIN" ]]; then
  # Windows (Git Bash, MSYS, Cygwin)
  echo "Windows detected. Please install the following manually:"
  echo "- GDAL (https://gdal.org/download.html or via Conda: conda install -c conda-forge gdal)"
  echo "- Make and GCC (optional, for advanced features)"
  echo "- Python 3 and pip"
  exit 0
else
  echo "Unsupported OS: $OS"
  exit 1
fi

echo "Setup complete!"