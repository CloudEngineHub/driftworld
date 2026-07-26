"""
Main for evaluating visual quality metrics for DriftWorld's generation on Robomimic
"""

import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs/train", config_name="can_1view")
def main(cfg: "DictConfig"):
    log.info("eval metrics start")
    from eval.eval_on_many_videos import evaluate_on_many_videos
    evaluate_on_many_videos(cfg, step=288500)
    log.info("eval metrics done")

if __name__ == "__main__":
    main()