"""
Evaluate visual quality metrics for DriftWorld on Robomimic.
"""
import os
import sys
import time
import logging

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import numpy as np
import torch

from eval.vis import set_seed, setup_model, _make_full_episode_dataset, _load_episode, _cameras_for_cfg
from data.create_robomimic_loader import AGENTVIEW_CAMERA, WRIST_CAMERA
from eval.eval_metrics import get_ssim, get_psnr, get_lpips

log = logging.getLogger(__name__)


_VIEW_NAMES = {AGENTVIEW_CAMERA: "agv", WRIST_CAMERA: "wrist"}


def _views_for_cfg(cfg):
    cameras = _cameras_for_cfg(cfg)
    if len(cameras) == 1:
        return [("", slice(0, 3))]
    return [(_VIEW_NAMES[cam], slice(3 * i, 3 * i + 3)) for i, cam in enumerate(cameras)]


def _view_key(name, metric):
    return f"{name}_{metric}" if name else metric


def evaluate_on_many_videos(cfg, step=None, split="valid"):
    """
    Evaluate visual quality metrics for DriftWorld on generated videos of the full Robomimic episodes.

    Args:
        cfg: Hydra config
        step: checkpoint step to load (None = latest checkpoint).
        split: name of the dataset split to use, e.g. 'valid'
    Returns:
        summary dict of per-view metric means plus generation timing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Evaluation of DriftWorld on the '{split}' split")
    set_seed(cfg.train.seed)

    cameras = _cameras_for_cfg(cfg)
    views = _views_for_cfg(cfg)

    # Full-episode dataset
    dataset = _make_full_episode_dataset(cfg, split)
    demos = list(dataset.demos)
    log.info(f"dataset '{split}' split: evaluating {len(demos)} episodes")

    denoiser, actual_step = setup_model(cfg, step, device)
    denoiser = denoiser.to(device).eval()
    h = cfg.model.num_history_frames - 1  # index of the current frame o_t within the seed window
    n = cfg.model.num_future_frames       # frames generated per forward pass
    log.info(f"loaded denoiser step={actual_step} h={h} n={n}")

    # Metric accumulators: one list per (view, metric), plus timing totals.
    metrics = {_view_key(name, m): [] for name, _ in views for m in ("ssim", "psnr", "lpips")}
    total_gen_time = 0.0
    total_gen_frames = 0

    for ds_idx, demo_id in enumerate(demos):
        frames, actions, _ = _load_episode(dataset, ds_idx, cfg.data.new_size, cfg.data.normalize_img, cameras)  # (L,C,H,W), (L,7)
        L = frames.shape[0]

        # Make the predictable-frame count (L - h - 1) become a multiple of n
        n_chunks = (L - h - 1) // n
        if n_chunks <= 0:
            log.warning(f"skipping {demo_id}: len {L} too short for h={h}, n={n}")
            continue
        L_trim = (h + 1) + n_chunks * n
        n_gen = n_chunks * n

        cur_state = frames[:L_trim].unsqueeze(0).to(device)  # (1, L_trim, C, H, W)
        act = actions[:L_trim].unsqueeze(0).to(device)       # (1, L_trim, 7)
        log.info(f"({ds_idx+1}/{len(demos)}) {demo_id} len {L} -> trimmed {L_trim} | n_gen={n_gen}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        with torch.no_grad():
            out = denoiser.sample_autoregressive(cur_state, act)  # (1, L_trim, C, H, W)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - _t0

        # Generated region only
        gen_g = out[0, h+1 : L_trim].float().cpu()        # (n_gen, C, H, W) in [-1, 1]
        gt_g = cur_state[0, h+1 : L_trim].float().cpu()   # (n_gen, C, H, W) in [-1, 1]

        # Per-view metrics
        for name, sl in views:
            gen_v = gen_g[:, sl]
            gt_v = gt_g[:, sl]
            ssim_m = float(get_ssim(gen_v, gt_v).mean())
            psnr_m = float(get_psnr(gen_v, gt_v).mean())
            lpips_m = float(get_lpips(gen_v, gt_v).mean())
            metrics[_view_key(name, "ssim")].append(ssim_m)
            metrics[_view_key(name, "psnr")].append(psnr_m)
            metrics[_view_key(name, "lpips")].append(lpips_m)

            label = f" ({name})" if name else ""
            log.info(f"  [metrics]{label} {demo_id} L_trim={L_trim} n_gen={n_gen}: "
                     f"SSIM={ssim_m:.4f} PSNR={psnr_m:.4f} LPIPS={lpips_m:.4f}")
            log.info(f"  [running avg]{label} over {len(metrics[_view_key(name, 'ssim')])} videos: "
                     f"SSIM={np.mean(metrics[_view_key(name, 'ssim')]):.4f} "
                     f"PSNR={np.mean(metrics[_view_key(name, 'psnr')]):.4f} "
                     f"LPIPS={np.mean(metrics[_view_key(name, 'lpips')]):.4f}")

        total_gen_time += gen_time
        total_gen_frames += n_gen
        log.info(f"  {demo_id} gen_time={gen_time:.2f}s")

    tpf = (total_gen_time / total_gen_frames) if total_gen_frames > 0 else float('nan')
    log.info(f"[timing] FINAL: total gen_time={total_gen_time:.1f}s over "
             f"{total_gen_frames} generated frames | time/frame={tpf:.4f}s")

    n_scored = len(metrics[_view_key(views[0][0], "ssim")])
    if n_scored == 0:
        log.info("[summary] no videos were evaluated")
        return None

    summary = {"n_videos": n_scored, "actual_step": actual_step}
    for name, _ in views:
        label = f" ({name})" if name else ""
        parts = []
        for m in ("ssim", "psnr", "lpips"):
            vals = np.asarray(metrics[_view_key(name, m)], dtype=np.float64)
            mean = float(vals.mean())
            summary[_view_key(name, m)] = mean
            parts.append(f"{m.upper()}={mean:.4f}")
        log.info(f"[summary]{label} per-video averages over {n_scored} videos "
                 f"(actual_step={actual_step}): " + " ".join(parts))

    summary["total_gen_time_s"] = total_gen_time
    summary["total_gen_frames"] = total_gen_frames
    summary["time_per_frame_s"] = tpf
    return summary
