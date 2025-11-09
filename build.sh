#!/bin/bash
# Build script for Render.com deployment
# Installs system dependencies and Python packages

set -e  # Exit on any error

echo "Starting build process..."

# Update package list
echo "Updating package list..."
apt-get update -qq

# Install system dependencies
echo "Installing system dependencies..."
apt-get install -y -qq \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libpoppler-cpp-dev \
    > /dev/null 2>&1

# Verify Tesseract installation
echo "Verifying Tesseract installation..."
tesseract --version || echo "Warning: Tesseract installation check failed"

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements-deploy.txt

# Create database directory if using persistent disk
if [ -n "$DATABASE_PATH" ]; then
    DB_DIR=$(dirname "$DATABASE_PATH")
    if [ "$DB_DIR" != "." ] && [ "$DB_DIR" != "" ]; then
        echo "Creating database directory: $DB_DIR"
        mkdir -p "$DB_DIR"
    fi
fi

echo "Build completed successfully!"
