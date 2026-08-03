"""
Utilities for evaluation
"""
import numpy as np
import torch
import os
import logging
import time
import random

from utils.model_utils import create_model, load_denoiser_state_dict
from utils.video_utils import write_numpy_to_mp4
from eval.tensor_conversions import convert_to_uint8_np

log = logging.getLogger(__name__)

def set_seed(seed):
    if seed == -1:
        seed = int(time.time())
    log.info(f"Seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed

def save_video(video_tensor, name, fps, value_range):
    """
    video_tensor: [Frames, Channels, Height, Width]
    name: filename for the video, must end in .mp4
    fps: fps for saved video
    value_range: either (0, 1) or (-1, 1)
    """
    numpy_vid = convert_to_uint8_np(video_tensor, value_range)
    numpy_vid = np.transpose(numpy_vid, (0, 2, 3, 1))
    write_numpy_to_mp4(numpy_vid, name, fps=fps)

def frames_from_video(video):
    """Raw (B, T, H, W, C) in [0,1] -> channel-first (B, T, C, H, W) in [-1,1]."""
    return video.permute(0, 1, 4, 2, 3).contiguous().mul(2.0).sub(1.0)

def setup_model(cfg, step):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Creating model")
    denoiser = create_model(cfg, device)
    
    log.info("Restoring ckpt")
    filepath = f"{cfg.output_dir}/ckpt_save/ckpt-latest.pth" if step is None else f"{cfg.output_dir}/ckpt_save/ckpt-step{step}.pth"
    if os.path.exists(filepath):
        ckpt = torch.load(filepath, map_location=device, weights_only=False)
        load_denoiser_state_dict(denoiser, ckpt['model'])
        actual_step = ckpt.get('step', step)
        del ckpt
        log.info(f"Restored from step {actual_step} ckpt")
    else:
        log.info(f"Checkpoint {filepath} does not exist")
        raise ValueError(f"Checkpoint {filepath} does not exist")

    return denoiser, actual_step

def gen_helper(denoiser, batch, use_ema, use_autoregressive = False, cfg_scale = None):
    """
    Generate from the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'video_latents' in batch:
        obs = batch['video_latents'].to(device)
    else:
        obs = frames_from_video(batch['video'].to(device))
    act = batch['action'].to(device)
    log.info(f"obs: {obs.shape}")
    log.info(f"act: {act.shape}")

    cfg_kwargs = {} if cfg_scale is None else {"cfg_scale": cfg_scale}
    if cfg_scale is not None:
        log.info(f"Sampling with cfg_scale={cfg_scale}")

    T_orig = obs.shape[1]
    if use_autoregressive:
        log.info(f"Use sample_autoregressive")
        gen = denoiser.sample_autoregressive(obs, act, use_ema=use_ema, init_val=None, gen_chunk=8, **cfg_kwargs)
    else:
        log.info("Use sample_not_autoregressive")
        gen = denoiser.sample_not_autoregressive(obs, act, use_ema=use_ema, init_val=None, **cfg_kwargs)

    gen = gen[:, :T_orig]
    log.info(f"gen: {gen.shape} | min {gen.min()} | max {gen.max()}")
    return gen.detach().cpu()
