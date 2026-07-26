# Download a robomimic raw demo hdf5 from Hugging Face
# Usage: bash download_raw_data.sh [task]
#   task:    lift | can

TASK="${1:-lift}"

URL="https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/main/v1.5/${TASK}/mh/demo_v15.hdf5"
OUT_DIR="./datasets/${TASK}/mh"
FILENAME="${OUT_DIR}/demo_v15.hdf5"

mkdir -p "$OUT_DIR"
echo "Downloading ${TASK}/mh/demo_v15.hdf5 from Hugging Face..."

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