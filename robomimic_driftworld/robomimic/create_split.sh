# Create a train / validation split
# Usage: bash create_split.sh [task]
#   task:    lift | can

TASK="${1:-lift}"
DIR="./datasets/${TASK}/mh"

python robomimic/scripts/split_train_val.py --dataset "${DIR}/image_v15.hdf5" --ratio 0.1
