# Extract image observations (both cameras, 84x84) from a raw demo hdf5.
# Usage: bash extract_obs_from_raw_data.sh [task]
#   task:    lift | can

TASK="${1:-lift}"
DIR="./datasets/${TASK}/mh"

python robomimic/scripts/dataset_states_to_obs.py --done_mode 2 \
--dataset "${DIR}/demo_v15.hdf5" \
--output_name image_v15.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
