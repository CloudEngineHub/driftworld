"""
Main for training DriftWorld. Examples:
torchrun --nproc_per_node=2 main_train.py --config-name=can_1view
torchrun --nproc_per_node=2 main_train.py --config-name=lift_1view
"""

import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="can_1view")
def main(cfg: DictConfig):
    log.info("Main start")
    from train import train
    train(cfg)
    log.info("Main done")

if __name__ == "__main__":
    main()