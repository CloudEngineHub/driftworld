<div align="center">
<h2 style="margin-bottom: 0.3em;">DriftWorld: Fast World Modeling through Drifting</h2>
<p style="margin: 0.3em 0;">Susie Lu, Haonan Chen, Weirui Ye, Yilun Du</p>
<p style="margin: 0.3em 0;"><a href="https://arxiv.org/abs/2607.15065">Paper</a> | <a href="https://susie-lu.github.io/driftworld/">Project Page</a> </p>
</div>

This codebase contains the official implementation for the DriftWorld paper.

<img src="assets/Teaser.png" width="700px"/>

## Setup and Checkpoints

You can install the relevant libraries by running `conda env create -f environment.yml`. DriftWorld checkpoints can be downloaded at [this HuggingFace link](https://huggingface.co/Susie-Lu/driftworld).

| Dataset | Code Folder |
|---|---|
| Push-T | [`pusht_driftworld`](https://github.com/Susie-Lu/driftworld/tree/main/pusht_driftworld) |
| Robomimic | [`robomimic_driftworld`](https://github.com/Susie-Lu/driftworld/tree/main/robomimic_driftworld) |
| RT-1 | [`rt1_driftworld`](https://github.com/Susie-Lu/driftworld/tree/main/rt1_driftworld) |
| Bridge-V2 | [`bridge_driftworld`](https://github.com/Susie-Lu/driftworld/tree/main/bridge_driftworld) |
| Language Table | [`language_table_driftworld`](https://github.com/Susie-Lu/driftworld/tree/main/language_table_driftworld) |

## DriftWorld on Push-T
The folder `pusht_driftworld` contains the code for training and evaluating DriftWorld on Push-T. Please put the pretrained checkpoints in the folder `pusht_driftworld/pusht_checkpoints`, and download the [dataset](https://huggingface.co/datasets/han2019/gpc_pushT_data/tree/main/world_model_data) in the folder `pusht_driftworld/pusht_data`.

**To train DriftWorld:**
- Run `torchrun --nproc_per_node=2 main_train.py --config-name=pushT_driftworld`. Experiments were run on 2 H100 GPUs.

**To visualize generated videos:**
- Run `python main_vis.py` to generate videos using DriftWorld.

**To evaluate visual quality metrics:**
- Run `python main_eval_metrics.py`. This will compute the MSE, SSIM, PSNR, and LPIPS metrics on the generated videos.

**To evaluate DriftWorld's performance on policy improvement:**
- Run `main_gpc_rank.py` using the instructions in the file. This will compute the IoU score of a baseline policy after applying inference-time policy improvement by rolling out action proposals in DriftWorld and selecting the best ones.

**To evaluate DriftWorld's performance on policy evaluation:** 
- Run `python main_policy_eval.py`. This will compute the IoU scores of the policy when rolled out in DriftWorld, compared to the ground-truth IoU scores.

## DriftWorld on Robomimic
The folder `robomimic_driftworld` contains the code for training and evaluating DriftWorld on the Robomimic tasks. Please put the pretrained checkpoints in the folder `robomimic_driftworld/robomimic_checkpoints`, and set up the dataset by following [this](https://github.com/Susie-Lu/driftworld/blob/main/robomimic_driftworld/robomimic/README.md).

**To train DriftWorld:**
- Run `torchrun --nproc_per_node=2 main_train.py --config-name=<config name>` with one of the configs in `robomimic_driftworld/configs/train`.

**To visualize generated videos:**
- Run `python main_vis.py` to generate videos using DriftWorld.

**To evaluate visual quality metrics:**
- Run `python main_eval_metrics.py`. This will compute the SSIM, PSNR, and LPIPS metrics on the generated videos.

## DriftWorld on RT-1
The folder `rt1_driftworld` contains the code for training and evaluating DriftWorld on the RT-1 dataset. Please put the pretrained checkpoint in the folder `rt1_driftworld/rt1_checkpoints`, and set up the dataset by following [this](https://github.com/Susie-Lu/driftworld/blob/main/rt1_driftworld/datasets/README.md).

**To train DriftWorld:**
- Run `torchrun --nproc_per_node=2 main_train.py --config-name=rt1_phase1` for phase 1 training, and then run `torchrun --nproc_per_node=2 main_train.py --config-name=rt1_phase2` for phase 2 self-forcing training. Experiments were run on 2 H200 GPUs.

**To visualize generated videos:**
- Run `python main_vis.py` to generate videos using DriftWorld.

**To evaluate visual quality metrics:**
- Run `script/eval_on_validation.sh`. This will compute the SSIM, PSNR, LPIPS, FID, and FVD metrics on the generated videos.

**About the pretrained checkpoint:**
- On RT-1, DriftWorld uses DINOv3 and the SD3 VAE during training, and it uses the SD3 VAE during inference. These two need to be downloaded separately. The pretrained DriftWorld checkpoint does not contain DINOv3 or SD3 VAE.

## DriftWorld on Bridge-V2
The folder `bridge_driftworld` contains the code for training and evaluating DriftWorld on the Bridge-V2 dataset. Please put the pretrained checkpoint in the folder `bridge_driftworld/bridge_checkpoints`, and set up the dataset by following [this](https://github.com/Susie-Lu/driftworld/blob/main/bridge_driftworld/datasets/README.md).

**To train DriftWorld:**
- Run `torchrun --nproc_per_node=2 main_train.py --config-name=bridge_phase1` for phase 1 training, and then run `torchrun --nproc_per_node=2 main_train.py --config-name=bridge_phase2` for phase 2 self-forcing training. Experiments were run on 2 H200 GPUs.

**To visualize generated videos:**
- Run `python main_vis.py` to generate videos using DriftWorld.

**To evaluate visual quality metrics:**
- Run `script/eval_on_validation.sh`. This will compute the SSIM, PSNR, LPIPS, FID, and FVD metrics on the generated videos.

**About the pretrained checkpoint:**
- On Bridge-V2, DriftWorld uses DINOv3 and the SD3 VAE during training, and it uses the SD3 VAE during inference. These two need to be downloaded separately. The pretrained DriftWorld checkpoint does not contain DINOv3 or SD3 VAE.

## DriftWorld on Language Table
The folder `language_table_driftworld` contains the code for training and evaluating DriftWorld on the Language Table dataset. Please put the pretrained checkpoint in the folder `language_table_driftworld/language_table_checkpoints`, and set up the dataset by following [this](https://github.com/Susie-Lu/driftworld/blob/main/language_table_driftworld/dataset/README.md).

**To train DriftWorld:**
- Run `torchrun --nproc_per_node=2 main_train.py --config-name=language_table_phase1` for phase 1 training, and then run `torchrun --nproc_per_node=2 main_train.py --config-name=language_table_phase2` for phase 2 self-forcing training. Experiments were run on 2 H200 GPUs.

**To visualize generated videos:**
- Run `python main_vis.py` to generate videos using DriftWorld.

**To evaluate visual quality metrics:**
- Run `script/eval_on_validation.sh`. This will compute the SSIM, PSNR, LPIPS, FID, and FVD metrics on the generated videos.

## Contact
If you have any questions, feel free to contact me at `susielu [dot] research [at] gmail [dot] com`.

## Citation

If you find this work useful in your research, please consider citing:
```bib
@article{lu2026driftworld,
  title={DriftWorld: Fast World Modeling through Drifting},
  author={Lu, Susie and Chen, Haonan and Ye, Weirui and Du, Yilun},
  journal={arXiv preprint arXiv:2607.15065},
  year={2026}
}
```
