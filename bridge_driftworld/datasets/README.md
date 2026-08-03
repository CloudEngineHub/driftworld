## To set up Bridge-V2 dataset
- First, download the Bridge-V2 dataset from https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/
- Then, run `download_data.py` to process the TFDS dataset into the mp4 + npz + txt form. You can complete this processing on a single machine via
```bash
python download_data.py \
    --dataset_name bridge_v2 \
    --dataset_home "./bridge/raw" \
    --output_dir   "./bridge/processed" \
    --fps 10 \
    --splits train
```
You can also speed up the process by splitting the processing among several machines, e.g.
```bash
# Machine 0
python download_data.py --dataset_name bridge_v2 \
  --dataset_home "./bridge/raw" \
  --output_dir   "./bridge/processed" \
  --fps 10 --splits train \
  --num_shards 3 --shard_id 0

# Similarly run with --shard_id 1 and 2 on machines 1 and 2.

# After ALL finish — run once on any of the machines:
python download_data.py --dataset_name bridge_v2 \
  --output_dir "./bridge/processed" \
  --splits train --finalize_manifest
```
