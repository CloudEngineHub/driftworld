"""
Visualize DriftWorld's generated videos on Robomimic.
"""
import os
import sys
import json
import random
import logging
from pathlib import Path

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import numpy as np
import torch
import imageio.v2 as imageio

from data.create_robomimic_loader import (
    _init_obs_utils,
    _resize_frames,
    AGENTVIEW_CAMERA,
    CAMERAS,
)
from robomimic.utils.dataset import SequenceDataset
from utils_model import create_model
from eval.tensor_conversions import convert_to_uint8_np

log = logging.getLogger(__name__)


def _cameras_for_cfg(cfg):
    """
    Returns the camera views used by the model
    """
    return CAMERAS if cfg.model.unet_type == "multi_2view" else (AGENTVIEW_CAMERA,)


def set_seed(seed):
    log.info(f"Seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


def _make_single_episode_dataset(hdf5_path, cameras, hdf5_cache_mode, split):
    """
    Build one full-episode (frame_stack=1, seq_length=1) SequenceDataset for a single hdf5.
    """
    return SequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=cameras,
        action_keys=("actions",),
        dataset_keys=("actions",),
        action_config={"actions": {"normalization": None}},
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=False,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        load_next_obs=False,
        filter_by_attribute=split,
    )


def _make_full_episode_dataset(cfg, split):
    """
    Build the full-episode dataset (frame_stack=1, seq_length=1).
    """
    cameras = _cameras_for_cfg(cfg)
    _init_obs_utils(cameras)
    return _make_single_episode_dataset(cfg.data.hdf5_path, cameras, cfg.data.hdf5_cache_mode, split)


def _process_view(frames, new_size, normalize_img):
    """
    Resize/normalize/permute a (T, H, W, 3) uint8 array to (T, 3, H, W) float32.
    """
    if new_size is not None:
        frames = _resize_frames(frames, new_size)
    frames = frames.astype(np.float32)
    if normalize_img:
        frames = frames / 255.0
        frames = (frames - 0.5) / 0.5
    frames = np.transpose(frames, (0, 3, 1, 2))
    return frames


def _load_episode(dataset, idx, new_size, normalize_img, cameras=(AGENTVIEW_CAMERA,)):
    """
    Load a full episode at dataset index idx.
    When there are multiple cameras views, they are concatenated channel-wise in the specified order.

    Returns:
        frames: (T, 3*len(cameras), H, W) float32 torch tensor (normalized to [-1, 1] if normalize_img)
        actions: (T, 7) float32 torch tensor
        demo_id: str, e.g. "demo_3"
    """
    traj = dataset.get_trajectory_at_index(idx)
    views = [_process_view(traj["obs"][cam], new_size, normalize_img) for cam in cameras]
    frames = np.concatenate(views, axis=1)
    actions = np.asarray(traj["actions"], dtype=np.float32)
    demo_id = traj["ep"]
    return torch.from_numpy(np.ascontiguousarray(frames)), torch.from_numpy(actions), demo_id


def setup_model(cfg, step, device):
    """
    Create the model and restore weights from a checkpoint.
    """
    log.info("Creating model")
    denoiser = create_model(cfg, device)

    filepath = f"{cfg.output_dir}/ckpt_save/ckpt-latest.pth" if step is None else f"{cfg.output_dir}/ckpt_save/ckpt-step{step}.pth"
    if not os.path.exists(filepath):
        raise ValueError(f"Checkpoint {filepath} does not exist")

    log.info(f"Restoring ckpt from {filepath}")
    ckpt = torch.load(filepath, map_location=device, weights_only=True)
    denoiser.load_state_dict(ckpt['model'])
    actual_step = ckpt.get('step', step)
    del ckpt
    denoiser.eval()
    log.info(f"Restored from step {actual_step} ckpt")
    return denoiser, actual_step


def _to_side_by_side(video):
    views = torch.split(video, 3, dim=1)  # V tensors of (T, 3, H, W)
    return torch.cat(views, dim=3)  # (T, 3, H, V*W)


def save_video(video_tensor, path, fps, value_range=(-1., 1.)):
    """
    Save a (T, 3*V, H, W) float tensor as an mp4.

    Args:
        video_tensor: (T, 3*V, H, W) float tensor with values in value_range
        path: output filename ending in .mp4
        fps: frames per second
        value_range: (0, 1) or (-1, 1)
    """
    combined = _to_side_by_side(video_tensor)  # (T, 3, H, V*W)
    frames = convert_to_uint8_np(combined, value_range)  # (T, 3, H, V*W) uint8
    frames = np.transpose(frames, (0, 2, 3, 1))  # (T, H, V*W, C)
    imageio.mimsave(path, list(frames), fps=fps)


@torch.no_grad()
def vis_on_validation(cfg, n_videos=64, step=None, fps=15, split="valid",
                      selection_file=None, save_gt=True):
    """
    Generate full-length videos for a fixed subset of the validation set and save them as mp4s.

    Args:
        cfg: Hydra config.
        n_videos: number of validation episodes to visualize
        step: checkpoint step to load (None = latest checkpoint)
        fps: frames per second for saved mp4s
        split: dataset filter_by_attribute key
        selection_file: path to the JSON file recording the selected demo_ids. If None, a fixed
            path next to this file is used so every run/checkpoint reuses the same subset.
        save_gt: if True, also save the ground-truth mp4 for each episode
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cameras = _cameras_for_cfg(cfg)
    view_tag = "_2view" if len(cameras) > 1 else ""
    log.info(f"Full-episode visualization on the '{split}' split "
             f"(n_videos={n_videos}, num_views={len(cameras)})")
    set_seed(cfg.train.seed)

    dataset = _make_full_episode_dataset(cfg, split)
    demos = list(dataset.demos)
    demo_to_idx = {d: i for i, d in enumerate(demos)}

    # selection for which episodes to generate
    if selection_file is None:
        selection_file = str(Path(__file__).resolve().parent /
                             f"vis_on_validation_selection_{split}_{n_videos}.json")

    if os.path.exists(selection_file):
        with open(selection_file, "r") as f:
            selected_demos = json.load(f)
        log.info(f"Loaded {len(selected_demos)} demo_ids from {selection_file}")
    else:
        selected_demos = demos[:min(n_videos, len(demos))]
        os.makedirs(os.path.dirname(selection_file), exist_ok=True)
        with open(selection_file, "w") as f:
            json.dump(selected_demos, f, indent=2)
        log.info(f"Selected first {len(selected_demos)} demo_ids and saved them to {selection_file}")

    denoiser, actual_step = setup_model(cfg, step, device)

    folder_root = f"{cfg.output_dir}/vis{view_tag}"
    os.makedirs(folder_root, exist_ok=True)
    gt_root = f"{folder_root}/gt"
    if save_gt:
        os.makedirs(gt_root, exist_ok=True)

    h = cfg.model.num_history_frames - 1  # index of the current frame o_t within the window
    n = cfg.model.num_future_frames       # frames generated per forward pass

    processed = 0
    for demo_id in selected_demos:
        if demo_id not in demo_to_idx:
            log.warning(f"Selected demo '{demo_id}' not found in the current '{split}' dataset; skipping")
            continue

        frames, actions, _ = _load_episode(
            dataset, demo_to_idx[demo_id], cfg.data.new_size, cfg.data.normalize_img, cameras)
        L = frames.shape[0]

        # Make the predictable-frame count (L - h - 1) become a multiple of n
        n_chunks = (L - h - 1) // n
        if n_chunks <= 0:
            log.warning(f"Episode '{demo_id}' (len {L}) is too short for h={h}, n={n}; skipping")
            continue
        L_trim = (h + 1) + n_chunks * n

        cur_state = frames[:L_trim].unsqueeze(0).to(device)  # (1, L_trim, C, H, W)
        act = actions[:L_trim].unsqueeze(0).to(device)       # (1, L_trim, 7)

        out = denoiser.sample_autoregressive(cur_state, act)  # (1, L_trim, C, H, W)
        gen_path = f"{folder_root}/{demo_id}_step{actual_step}_len{L_trim}.mp4"
        save_video(out[0].cpu(), gen_path, fps=fps, value_range=(-1, 1))
        log.info(f"saved video at {gen_path}")

        if save_gt:
            gt_path = f"{gt_root}/{demo_id}_gt_len{L_trim}.mp4"
            if os.path.exists(gt_path):
                log.info(f"GT video already exists, skipping: {gt_path}")
            else:
                save_video(cur_state[0].cpu(), gt_path, fps=fps, value_range=(-1, 1))
                log.info(f"saved ground-truth video at {gt_path}")

        processed += 1

    log.info(f"Done. Visualized {processed} episodes under {folder_root}")
