"""
Main for training DriftWorld. Example for launching training run:
torchrun --nproc_per_node=2 main_train.py --config-name=language_table_phase1
"""
import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs/train", config_name="")
def main(cfg: DictConfig):
    log.info("Main start")
    if not cfg.model.is_phase_2:
        from train import train
        train(cfg)
    else:
        from train_selfforce import train
        train(cfg)
    log.info("Main done")

if __name__ == "__main__":
    main()