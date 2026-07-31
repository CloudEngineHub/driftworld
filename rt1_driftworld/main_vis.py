"""
Main for visualizing generated videos
"""

import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs/sample", config_name="rt1_release")
def main(cfg: DictConfig):
    log.info("Main start")

    from eval.vis_on_validation import vis_on_validation

    step = 29400
    vis_on_validation(
        cfg,
        n_videos=128,
        step=step,
        fps=6,
        split="val",
        chunk_size=16,
        batch_size=2,
        cfg_scale=2.5,
        selection_seed=0,
        selection_file=None,
    )

    log.info("Main done")

if __name__ == "__main__":
    main()
