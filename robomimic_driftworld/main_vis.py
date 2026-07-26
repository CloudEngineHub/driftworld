"""
Main for visualizing (generating + saving) DriftWorld rollouts on Robomimic tasks.
"""

import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs/train", config_name="lift_1view")
def main(cfg: DictConfig):
    log.info("Main start")
    from eval.vis import vis_on_validation
    vis_on_validation(cfg,
                      n_videos=30,
                      step=212000,
                      fps=15,
                      split="valid",
                      selection_file=None,
                      save_gt=True)
    log.info("Main done")

if __name__ == "__main__":
    main()
