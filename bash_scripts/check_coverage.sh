#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define the project's root directory relative to the script's location
# Assuming the script is in a subdirectory of the project root
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define the report path relative to the project root
REPORT_PATH="$PROJECT_ROOT/tests"

# Navigate to the project root
cd "$PROJECT_ROOT"

# Run coverage
coverage run -m unittest discover -s tests

# Generate a coverage report and save it
coverage report > "$REPORT_PATH/coverage_report.txt"
# Uncomment the line below if you prefer an HTML report
# coverage html -d "$REPORT_PATH/htmlcov"

echo "Coverage report saved to $REPORT_PATH"
