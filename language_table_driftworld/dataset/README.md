## To set up the Language Table dataset

First, download the Language Table dataset from this [HuggingFace link](https://huggingface.co/datasets/IPEC-COMMUNITY/language_table_lerobot), and put it at the location `./language_table` (or any other location - just change the dataset filepath in the config files).

To create a validation set, randomly select 2500 videos (that are sufficiently long) from the dataset. The file `./language_table/valid_videos_2500.json` is an example selection of 2500 videos.
