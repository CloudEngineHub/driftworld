## To set up the RT-1 dataset
- First, run `download_rt1.sh` to download the raw TFDS dataset.
- Then, run `download_data.py` to process the TFDS dataset into the mp4 + npz + txt form:
```bash
python download_data.py \
    --dataset_name rt_1 \
    --dataset_home "./rt1/raw_tfds" \
    --output_dir   "./rt1/processed" \
    --fps 3 \
    --val_holdout_frac 0.025 \
    --val_seed 314
```
