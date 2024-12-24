#!/bin/bash

# Create directories
mkdir -p small tiny medium base

# Move files to appropriate directories
for file in results_question_*.wav_ggml-*.en_*.txt; do
    if [[ $file == *"ggml-small"* ]]; then
        mv "$file" small/
    elif [[ $file == *"ggml-tiny"* ]]; then
        mv "$file" tiny/
    elif [[ $file == *"ggml-medium"* ]]; then
        mv "$file" medium/
    elif [[ $file == *"ggml-base"* ]]; then
        mv "$file" base/
    fi
done

echo "Files have been sorted into their respective folders."
