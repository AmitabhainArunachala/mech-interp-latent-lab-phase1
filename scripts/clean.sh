#!/bin/bash
# clean.sh - Remove all Python cache files and temporary artifacts
# Usage: ./clean.sh

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "✅ Removed __pycache__ directories"

# Remove .pyc files
find . -type f -name "*.pyc" -delete
echo "✅ Removed .pyc files"

# Remove .pytest_cache if it exists
if [ -d ".pytest_cache" ]; then
    rm -rf .pytest_cache
    echo "✅ Removed .pytest_cache"
fi

# Remove .mypy_cache if it exists
if [ -d ".mypy_cache" ]; then
    rm -rf .mypy_cache
    echo "✅ Removed .mypy_cache"
fi

echo "✨ Workspace clean."









