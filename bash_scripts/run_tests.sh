#!/bin/bash

# Navigate to the parent directory (repo root) of the script's directory
cd "$(dirname "$0")/.."

# Activate your environment if you have one (optional)
# source path_to_your_env/bin/activate

# Run pytest on the tests folder
python -m unittest discover -s tests -p 'test_*.py'

# Deactivate your environment if you activated it (optional)
# deactivate