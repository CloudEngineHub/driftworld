"""
Language Table dataloader
"""
import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .language_table_dataset import LanguageTablePrecomputedDataset

log = logging.getLogger(__name__)


def _build_dataset(cfg, split, *, full_video=False):
    return LanguageTablePrecomputedDataset(
        dataset_root=cfg.language_table.dataset_root,
        n_frames=cfg.language_table.n_frames,
        frame_skip=cfg.language_table.get("frame_skip", 1),
        action_dim=cfg.language_table.get("action_dim", 2),
        action_scale=cfg.language_table.get("action_scale", 20.0),
        input_h=cfg.language_table.get("input_h", 144),
        input_w=cfg.language_table.get("input_w", 256),
        split=split,
        val_frac=cfg.language_table.get("val_frac", 0.0),
        val_episodes_file=cfg.language_table.get("val_episodes_file", None),
        max_retries=cfg.language_table.get("max_retries", 100),
        full_video=full_video,
    )


def _pad_collate_full_video(batch):
    lengths = [item["video"].shape[0] for item in batch]
    t_max = max(lengths)

    def pad_time(x):
        pad_t = t_max - x.shape[0]
        if pad_t == 0:
            return x
        pad = torch.zeros((pad_t, *x.shape[1:]), dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    out = {
        "video": torch.stack([pad_time(item["video"]) for item in batch], dim=0),
        "action": torch.stack([pad_time(item["action"]) for item in batch], dim=0),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
    return out


def get_language_table_eval_dataloader(cfg, split="val", batch_size=4, num_workers=None):
    """
    Returns a dataloader that yields FULL Language Table episodes, padded per batch.
    """
    log.info(f"Initializing Language Table {split} FULL-VIDEO eval dataset")
    dataset = _build_dataset(cfg, split, full_video=True)

    if num_workers is None:
        num_workers = cfg.dataloader.num_workers

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_pad_collate_full_video,
        pin_memory=cfg.dataloader.pin_memory,
    )
    log.info(f"Created {split} full-video eval dataloader with {len(dataset)} videos "
             f"(batch_size={batch_size}, batches={len(dataloader)}).")
    return dataloader


def get_language_table_dataloader(cfg, split, rank=0, world_size=1):
    """
    Returns a dataloader for Language Table and the specified split.
    Args:
        cfg: Hydra config
        split (str): "train" or "val"
        rank: rank of this process (0 if single-GPU)
        world_size: total number of distributed processes (1 if single-GPU)
    """
    log.info(f"Initializing Language Table {split} dataset")
    dataset = _build_dataset(cfg, split)
    log.info(f"Initializing Language Table {split} dataset: done")

    batch_size = cfg.dataloader.batch_size if split == "train" else 1

    sampler = None
    shuffle = (split == "train")
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=(split == "train"),
        )
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        shuffle=shuffle,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=True,
    )

    log.info(f"Created {split} dataloader with {len(dataset)} data points "
             f"(rank={rank}/{world_size}, per-rank-batches={len(dataloader)}).")
    return dataloader
