"""
RT-1 Dataloader
"""
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .dataset_precomputed import OpenXMP4PrecomputedDataset

log = logging.getLogger(__name__)


def _pad_collate_full_video(batch):
    """
    Collate variable-length full episodes into a padded batch.

    Each item is a dict with 
        - 'video': (T, H, W, C) pixel video
        - 'action': (T, action_dim) actions

    Time dimensions are zero-padded to the batch max so the model can run on the whole batch at once.
    The true per-sample lengths are returned in 'lengths' so callers can trim the padded tail.
    """
    lengths = [item['video'].shape[0] for item in batch]
    t_max = max(lengths)

    def pad_time(x):
        pad_t = t_max - x.shape[0]
        if pad_t == 0:
            return x
        pad = torch.zeros((pad_t, *x.shape[1:]), dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    out = {
        'video': torch.stack([pad_time(item['video']) for item in batch], dim=0),
        'action': torch.stack([pad_time(item['action']) for item in batch], dim=0),
        'lengths': torch.tensor(lengths, dtype=torch.long),
    }
    return out


def get_rt1_eval_dataloader(cfg, split="val", batch_size=4, num_workers=None):
    """
    Creates a dataloader that returns FULL RT-1 episodes, padded per batch.
    This is only used during evaluation.

    Args:
        cfg: Hydra config
        split: "train" or "val" (evaluation uses "val")
        batch_size: number of videos per batch (padded to the batch's max length)
        num_workers: override; defaults to cfg.dataloader.num_workers
    """
    log.info(f"Initializing RT-1 {split} FULL-VIDEO eval dataset")
    dataset = OpenXMP4PrecomputedDataset(
        save_dir=cfg.rt1.dataset_dir,
        n_frames=cfg.rt1.n_frames,
        action_dim=cfg.rt1.action_dim,
        subset_names=cfg.rt1.subset_names,
        split=split,
        input_h=cfg.rt1.input_h,
        input_w=cfg.rt1.input_w,
        full_video=True,
    )

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


def get_rt1_dataloader(cfg, split, rank=0, world_size=1):
    """
    Returns dataloader for RT-1 for the specified split.
    This is used during training.

    Args:
        cfg: Hydra config
        split: "train" or "val"
        rank: rank of this process (0 if single-GPU)
        world_size: world size (1 if single-GPU)
    """
    log.info(f"Initializing RT-1 {split} dataset")
    dataset = OpenXMP4PrecomputedDataset(
        save_dir=cfg.rt1.dataset_dir,
        n_frames=cfg.rt1.n_frames,
        action_dim=cfg.rt1.action_dim,
        subset_names=cfg.rt1.subset_names,
        split=split,
        input_h=cfg.rt1.input_h,
        input_w=cfg.rt1.input_w,
    )
    log.info(f"Initializing RT-1 {split} dataset: done")

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
        persistent_workers=True
    )

    log.info(f"Created {split} dataloader with {len(dataset)} data points (rank={rank}/{world_size}, per-rank-batches={len(dataloader)}).")
    return dataloader
