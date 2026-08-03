"""
Utilities for evaluation
"""
import numpy as np
import torch
import os
import logging
import random

from utils.model_utils import create_model
from utils.video_utils import write_numpy_to_mp4
from eval.tensor_conversions import convert_to_uint8_np

log = logging.getLogger(__name__)

def set_seed(seed):
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

def setup_model(cfg, step):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Creating model")
    denoiser = create_model(cfg, device)
    
    log.info("Restoring ckpt")
    filepath = f"{cfg.output_dir}/ckpt_save/ckpt-latest.pth" if step is None else f"{cfg.output_dir}/ckpt_save/ckpt-step{step}.pth"
    if os.path.exists(filepath):
        ckpt = torch.load(filepath, map_location=device, weights_only=False)
        denoiser.load_state_dict(ckpt['model'])
        actual_step = ckpt.get('step', step)
        del ckpt
        log.info(f"Restored from step {actual_step} ckpt")
    else:
        log.info(f"Checkpoint {filepath} does not exist")
        raise ValueError(f"Checkpoint {filepath} does not exist")

    return denoiser, actual_step

def gen_helper(denoiser, batch, use_ema, use_autoregressive = True):
    """
    Generate from the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs = batch['video_latents'].to(device)
    act = batch['action'].to(device)
    log.info(f"obs: {obs.shape}")
    log.info(f"act: {act.shape}")

    T_orig = obs.shape[1]
    if use_autoregressive:
        log.info(f"Use sample_autoregressive")
        gen = denoiser.sample_autoregressive(obs.clone(), act, use_ema, init_val=None)
    else:
        log.info("Use sample_not_autoregressive")
        gen = denoiser.sample_not_autoregressive(obs, act, use_ema, init_val=None)
    gen = gen[:, :T_orig]
    log.info(f"gen: {gen.shape} | min {gen.min()} | max {gen.max()}")
    return gen.detach().cpu()
