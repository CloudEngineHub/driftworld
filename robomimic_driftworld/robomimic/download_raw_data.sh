#!/bin/bash

URL="https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5/lift/mh/demo_v15.hdf5"
FILENAME="demo_v15.hdf5"

echo "Downloading $FILENAME from Hugging Face..."

if command -v curl &> /dev/null; then
    echo "Using curl"
    curl -L -o "$FILENAME" "$URL"

elif command -v wget &> /dev/null; then
    echo "Using wget"
    wget -O "$FILENAME" "$URL"

else
    echo "Error: No suitable download tool found."
    exit 1
fi

echo "Download complete: $FILENAME"