#!/bin/bash
# Force installation script for SABER with all dependencies

set -e  # Exit on error

echo "Installing SABER with all dependencies (force mode)"
echo "⚠️ This may override dependency conflicts"

# Upgrade pip for better dependency resolution
echo "Upgrading pip..."
pip install --upgrade pip

# # Method 1: Try standard installation first
# echo "Attempting standard installation..."
# if pip install saber-query[all]; then
#     echo "✅ Standard installation successful!"
#     exit 0
# fi

# echo "⚠️ Standard installation failed, trying force methods..."

# Method 2: Sequential installation (your successful approach)
echo "Trying sequential installation..."
pip install lotus-ai docetl --upgrade
# pip install palimpzest --force-reinstall
# pip install 'palimpzest>=0.7.10,<0.8.0' --force-reinstall
pip install 'palimpzest>=1.1.0' --force-reinstall
# pip install saber-query --force-reinstall
pip install -e . --force-reinstall

# Verify installation
echo "Verifying installation..."
python -c "
import lotus
import docetl
import palimpzest
import openai
import duckdb
import sqlglot
print('✅ All packages imported successfully!')
print('📋 Package versions:')
"

echo "🎉 Installation complete! All packages are working together."
echo "If you encounter runtime issues, consider using separate environments for each framework."
