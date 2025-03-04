#! /bin/bash

# Change permissions for STRM folder
chmod -R 755 ./SRTM1

# Update package list
apt-get update

# Install essential build tools including make and gcc
apt-get install -y build-essential make gcc

# Optionally install other dependencies (e.g., for elevation or other libraries)
apt-get install -y python3-dev libgdal-dev

# Clean up to reduce the image size
apt-get clean

pip install git+https://github.com/marceloprates/prettymaps.git