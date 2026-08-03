"""
Evaluate visual quality metrics on the Language Table validation set.
This file supports launching on multiple single-GPU machine to speed up the evaluation process.

Example: Launch
    python -m eval.eval_on_validation_shard \
        --config-path configs/sample/language_table_phase2 --step 42800 \
        --rank 0 --world-size 4
on one GPU and similarly launch with --rank 1 on another GPU.
After all ranks are done, run eval_on_validation_aggregate.py to aggregate the metrics.
"""
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from omegaconf import OmegaConf

from torch.utils.data import DataLoader, Subset

from dataset.language_table_dataloader import get_language_table_eval_dataloader, _pad_collate_full_video
from encoders.vae_utils import decode_latents
from encoders.vae_sd3 import VAE as SD3VAE
from utils.eval_utils import set_seed, save_video, setup_model
from .eval_metrics import get_ssim, get_psnr, get_lpips

log = logging.getLogger(__name__)


N_FVD_FRAMES = 16


def _build_run_root(cfg, ckpt_tag):
    name = f"{ckpt_tag}_autoregressive"
    return Path(cfg.output_dir) / "eval_on_validation_multigpu" / name


def _to_uint8_hwc(frame01: np.ndarray) -> np.ndarray:
    return (np.clip(frame01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _write_frames_flat(frames01: np.ndarray, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, f in enumerate(frames01):
        imageio.imwrite(str(out_dir / f"{stem}_f{idx:04d}.png"), _to_uint8_hwc(f))


def _write_frames_per_video(frames01: np.ndarray, out_dir: Path, stem: str) -> None:
    sub = out_dir / stem
    sub.mkdir(parents=True, exist_ok=True)
    for idx, f in enumerate(frames01):
        imageio.imwrite(str(sub / f"frame_{idx:02d}.png"), _to_uint8_hwc(f))


def _pixels_m11_to_hwc01_np(pix: torch.Tensor) -> np.ndarray:
    return (((pix.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).contiguous()
            .float().cpu().numpy())


def run_shard(cfg, step, rank, world_size, *, batch_size=4,
              fps=4, chunk_size=16):
    assert 0 <= rank < world_size, f"bad rank {rank} for world_size {world_size}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_prefix = "(auto)"
    log.info(f"{mode_prefix} [shard {rank}/{world_size}] step={step} batch_size={batch_size} "
             f"use_ema=True (autoregressive)")
    set_seed(cfg.train.seed + rank)

    base_dataloader = get_language_table_eval_dataloader(cfg, split="val", batch_size=1)
    dataset = base_dataloader.dataset
    n_total = len(dataset)
    my_indices = [i for i in range(n_total) if i % world_size == rank]
    stems_all = [f"ep{ep['episode_index']:06d}" for ep in dataset.samples]
    log.info(f"[shard {rank}/{world_size}] dataset size {n_total}, "
             f"this shard owns {len(my_indices)} clips")

    subset = Subset(dataset, my_indices)
    shard_loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=_pad_collate_full_video,
        pin_memory=cfg.dataloader.pin_memory,
    )

    denoiser, actual_step = setup_model(cfg, step)
    denoiser = denoiser.to(device).eval()
    n_history = denoiser.num_history_steps
    log.info(f"[shard {rank}/{world_size}] loaded denoiser step={actual_step} "
             f"n_history={n_history} (SD3 VAE)")

    sd3_vae = None
    if cfg.model.vae_type == "sd3":
        log.info("Using SD3 VAE (encoding video latents on the fly)")
        sd3_vae = SD3VAE().to(device)

    # Output dirs
    ckpt_tag = f"step{actual_step}"
    run_root = _build_run_root(cfg, ckpt_tag)
    pred_videos_dir = run_root / "pred_videos"
    pred_frames_dir = run_root / "pred_frames"
    gt_frames_dir = run_root / "gt_frames"
    pred_first16_dir = run_root / "pred_first16"
    gt_first16_dir = run_root / "gt_first16"
    shards_dir = run_root / "shards"
    for d in (pred_videos_dir, pred_frames_dir, gt_frames_dir,
              pred_first16_dir, gt_first16_dir, shards_dir):
        d.mkdir(parents=True, exist_ok=True)

    records = []
    total_gen_time = 0.0
    total_gen_frames = 0

    n_batches = len(shard_loader)
    cursor = 0
    for batch_i, batch in enumerate(shard_loader):
        lengths = batch['lengths']
        B = int(lengths.shape[0])
        batch_indices = my_indices[cursor:cursor + B]
        cursor += B
        stems_batch = [stems_all[ds_idx] for ds_idx in batch_indices]
        mp4_paths = [pred_videos_dir / f"{stem}.mp4" for stem in stems_batch]
        existing_stems = {r["stem"] for r in records}
        all_done = all(
            mp4_paths[j].exists() and stems_batch[j] in existing_stems
            for j in range(B)
        )
        if all_done:
            log.info(f"[shard {rank}/{world_size}] (batch {batch_i+1}/{n_batches}) "
                     f"all {B} clips already done, skipping")
            continue

        log.info(f"[shard {rank}/{world_size}] (batch {batch_i+1}/{n_batches}) "
                 f"B={B} stems={stems_batch} lengths={lengths.tolist()}")

        actions = batch['action'].to(device)           # (B, L_max, action_dim)
        if sd3_vae is not None:
            video_pix = batch['video'].to(device)      # (B, L_max, H, W, C) [0,1]
            model_input = sd3_vae.encode(video_pix)
        else:
            model_input = batch['video_latents'].to(device)   # (B, L_max, C, h, w)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        with torch.no_grad():
            gen = denoiser.sample_autoregressive(
                model_input.clone(), actions,
                use_ema=True, init_val=None,
            ) # (B, L_max, 3, H_pix, W_pix) in [-1, 1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - _t0
        per_sample_gen_time = gen_time / B

        # GT frames
        with torch.no_grad():
            gt_pixels = decode_latents(model_input, chunk_size=chunk_size)   # (B, L_max, 3, H, W) [-1, 1]

        batch_n_gen = 0
        for j in range(B):
            stem = stems_batch[j]
            mp4_path = mp4_paths[j]
            L = int(lengths[j])
            n_gen = L - n_history
            if n_gen <= 0:
                log.warning(f"  skipping {stem}: L={L} <= n_history={n_history}")
                continue

            # Trim padded tail
            gen_g = gen[j, n_history:L].float().cpu()
            gt_g = gt_pixels[j, n_history:L].float().cpu()
            ssim_m = float(get_ssim(gen_g, gt_g).mean())
            psnr_m = float(get_psnr(gen_g, gt_g).mean())
            lpips_m = float(get_lpips(gen_g, gt_g).mean())

            log.info(f"  {mode_prefix} [j={j}] {stem} L={L} n_gen={n_gen}: "
                     f"SSIM={ssim_m:.4f} "
                     f"PSNR={psnr_m:.4f} LPIPS={lpips_m:.4f}")

            # Save predicted MP4
            save_video(gen[j, :L].cpu(), str(mp4_path), fps=fps, value_range=(-1, 1))

            # Dump generated frames (exclude the n_history GT seed frames)
            gen_for_fid = _pixels_m11_to_hwc01_np(gen_g)
            gt_for_fid = _pixels_m11_to_hwc01_np(gt_g)
            _write_frames_flat(gen_for_fid, pred_frames_dir, stem)
            _write_frames_flat(gt_for_fid, gt_frames_dir, stem)

            if n_gen >= N_FVD_FRAMES:
                _write_frames_per_video(
                    gen_for_fid[:N_FVD_FRAMES], pred_first16_dir, stem)
                _write_frames_per_video(
                    gt_for_fid[:N_FVD_FRAMES], gt_first16_dir, stem)
                fvd_kept = N_FVD_FRAMES
            else:
                log.warning(f"  {stem}: n_gen={n_gen} < {N_FVD_FRAMES}; skipping FVD dump")
                fvd_kept = 0

            records.append({
                "stem": stem,
                "L": L,
                "n_gen": n_gen,
                "ssim": ssim_m,
                "psnr": psnr_m,
                "lpips": lpips_m,
                "gen_time_s": per_sample_gen_time,
                "fvd_kept_frames": fvd_kept,
            })
            batch_n_gen += n_gen

        total_gen_time += gen_time
        total_gen_frames += batch_n_gen
        log.info(f"  batch gen_time={gen_time:.2f}s ({per_sample_gen_time:.2f}s/sample) "
                 f"frames={batch_n_gen}")

    final_tpf = (total_gen_time / total_gen_frames) if total_gen_frames > 0 else float('nan')
    log.info(f"[shard {rank}/{world_size}] done. {len(records)} clips, "
             f"gen_time={total_gen_time:.1f}s frames={total_gen_frames} "
             f"time/frame={final_tpf:.4f}s")

    out_json = shards_dir / f"metrics_rank{rank}_of{world_size}.json"
    with open(out_json, "w") as f:
        json.dump({
            "rank": rank, "world_size": world_size,
            "step": step, "actual_step": actual_step,
            "use_ema": True,
            "use_autoregressive": True,
            "n_history": n_history,
            "decoder_type": "sd3",
            "n_clips": len(records),
            "total_gen_time_s": total_gen_time,
            "total_gen_frames": total_gen_frames,
            "videos": records,
        }, f, indent=2)
    (shards_dir / f"done_rank{rank}_of{world_size}.flag").touch()
    log.info(f"[shard {rank}/{world_size}] wrote {out_json}")


def _build_parser():
    p = argparse.ArgumentParser(description="Multi-GPU eval -- per-shard generation")
    p.add_argument("--config-path", required=True, help="Hydra/OmegaConf YAML")
    p.add_argument("--step", type=int, default=None,
                   help="Checkpoint step to load (default: latest)")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Number of val clips processed per sample_autoregressive call")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=16,
                   help="GT SD3 VAE decode chunk size")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    cfg = OmegaConf.load(args.config_path)
    run_shard(
        cfg, args.step, args.rank, args.world_size,
        batch_size=args.batch_size,
        fps=args.fps,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
