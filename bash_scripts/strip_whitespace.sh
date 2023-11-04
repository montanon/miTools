#!/bin/bash
find mitools -type f -name "*.py" -exec sed -i '' 's/ *$//' '{}' ';'
