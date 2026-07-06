#!/bin/sh
set -e

echo "Pre-building extraction cache..."
python3 prebuild_cache.py || echo "Cache pre-build skipped (Ollama may not be ready yet)"

echo "Starting dashboard..."
exec python3 app.py
