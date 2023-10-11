#!/bin/bash

# Check if a path was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 path_to_notebook.ipynb"
    exit 1
fi

# Check if the file exists
if [ ! -f "$1" ]; then
    echo "File not found!"
    exit 1
fi

# Use jupyter nbconvert to clear outputs
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$1"

# Check for errors in the above command
if [ $? -eq 0 ]; then
    echo "Outputs cleared successfully!"
else
    echo "Error in clearing outputs!"
    exit 1
fi
