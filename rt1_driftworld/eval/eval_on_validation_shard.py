"""
Evaluate visual quality metrics on the RT-1 validation set
This file supports launching on multiple single-GPU machine to speed up the evaluation process.

Example: Launch
    python -m eval.eval_on_validation_shard \
        --config-path configs/sample/rt1_release.yaml --step 29400 \
        --rank 0 --world-size 1 --cfg-scale 2.5 --chunk-gen-frames 8
You can set a world size > 1 to use more than 1 GPU.
After all ranks are done, run eval_on_validation_aggregate.py to aggregate the metrics.
"""
import json
import time
import shutil
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from omegaconf import OmegaConf

from datasets.rt1_dataloader import get_rt1_eval_dataloader
from encoders.vae_utils import decode_latents
from encoders.vae_sd3 import VAE as SD3VAE
from utils.eval_utils import set_seed, save_video, setup_model
from .eval_metrics import get_ssim, get_psnr, get_lpips

log = logging.getLogger(__name__)


N_FVD_FRAMES = 16


def _build_run_root(cfg, ckpt_tag, cfg_scale, mode):
    name = f"{ckpt_tag}_cfg{cfg_scale}_{mode}"
    return Path(cfg.output_dir) / "eval_on_validation_multigpu" / name


def _to_uint8_hwc(frame01: np.ndarray) -> np.ndarray:
    """(H, W, 3) float [0, 1] -> (H, W, 3) uint8."""
    return (np.clip(frame01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _write_frames_flat(frames01: np.ndarray, out_dir: Path, stem: str) -> None:
    """Write each frame to out_dir/{stem}_f{idx:04d}.png."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, f in enumerate(frames01):
        imageio.imwrite(str(out_dir / f"{stem}_f{idx:04d}.png"), _to_uint8_hwc(f))


def _write_frames_per_video(frames01: np.ndarray, out_dir: Path, stem: str) -> None:
    """Write each frame to out_dir/{stem}/frame_{idx:02d}.png."""
    sub = out_dir / stem
    sub.mkdir(parents=True, exist_ok=True)
    for idx, f in enumerate(frames01):
        imageio.imwrite(str(sub / f"frame_{idx:02d}.png"), _to_uint8_hwc(f))


def _pixels_m11_to_hwc01_np(pix: torch.Tensor) -> np.ndarray:
    """(N, 3, H, W) in [-1, 1] -> (N, H, W, 3) float in [0, 1] (numpy)."""
    return (((pix.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).contiguous()
            .float().cpu().numpy())


def _config_resolved(config_path) -> str:
    """Stable config identity for resume checks."""
    return str(Path(config_path).expanduser().resolve())


def _build_resume_metadata(config_path, rank, world_size, actual_step, *, cfg_scale,
                           n_history, fps, chunk_size, max_videos_per_shard,
                           chunk_gen_frames):
    return {
        "rank": int(rank),
        "world_size": int(world_size),
        "actual_step": int(actual_step),
        "config_resolved": _config_resolved(config_path),
        "cfg_scale": float(cfg_scale),
        "n_history": int(n_history),
        "fps": int(fps),
        "chunk_size": int(chunk_size),
        "max_videos_per_shard": max_videos_per_shard,
        "chunk_gen_frames": int(chunk_gen_frames),
    }


def _record_is_usable(record: dict, stem: str) -> bool:
    required = ("stem", "n_gen", "ssim", "psnr", "lpips",
                "gen_time_s", "fvd_kept_frames")
    if not isinstance(record, dict):
        return False
    if any(k not in record for k in required):
        return False
    if record["stem"] != stem:
        return False
    try:
        return int(record["n_gen"]) > 0
    except Exception:
        return False


def _load_resume_records(out_json: Path, metadata: dict, expected_stems: list):
    """Load saved per-video records if the file belongs to this exact shard command."""
    if not out_json.exists():
        return {}, "no existing shard JSON"

    try:
        with open(out_json, "r") as f:
            payload = json.load(f)
    except Exception as e:
        return {}, f"could not read existing shard JSON ({e})"

    for key, expected in metadata.items():
        if payload.get(key) != expected:
            return {}, (
                f"existing shard JSON metadata mismatch for {key}: "
                f"{payload.get(key)!r} != {expected!r}"
            )

    expected_set = set(expected_stems)
    records_by_stem = {}
    for record in payload.get("videos", []):
        stem = record.get("stem") if isinstance(record, dict) else None
        if stem not in expected_set:
            continue
        if not _record_is_usable(record, stem):
            continue
        records_by_stem[stem] = record

    return records_by_stem, f"loaded {len(records_by_stem)} resumable records"


def _ordered_records(records_by_stem: dict, stems: list) -> list:
    return [records_by_stem[s] for s in stems if s in records_by_stem]


def _record_totals(records: list):
    total_gen_time = sum(float(r.get("gen_time_s", 0.0)) for r in records)
    total_gen_frames = sum(int(r.get("n_gen", 0)) for r in records)
    return total_gen_time, total_gen_frames


def _running_summary(records: list) -> str:
    """Running averages over all records completed in this shard so far."""
    n = len(records)
    if n == 0:
        return "no clips yet"
    total_gen_time, total_gen_frames = _record_totals(records)
    tpf = total_gen_time / total_gen_frames if total_gen_frames > 0 else float('nan')
    n_fvd = sum(1 for r in records if int(r.get("fvd_kept_frames", 0)) >= N_FVD_FRAMES)

    def _mean(key):
        return sum(float(r[key]) for r in records) / n

    return (f"n={n} | SSIM={_mean('ssim'):.4f} "
            f"PSNR={_mean('psnr'):.4f} LPIPS={_mean('lpips'):.4f} | "
            f"time/frame={tpf:.4f}s | fvd-eligible={n_fvd}/{n}")


def _build_shard_payload(metadata: dict, records_by_stem: dict, stems: list,
                         *, config_path, step, total_available_episodes, complete: bool):
    records = _ordered_records(records_by_stem, stems)
    total_gen_time, total_gen_frames = _record_totals(records)
    payload = dict(metadata)
    payload.update({
        "config": str(config_path),
        "step": step,
        "n_expected_clips": len(stems),
        "total_available_episodes": int(total_available_episodes),
        "complete": bool(complete),
        "n_clips": len(records),
        "total_gen_time_s": total_gen_time,
        "total_gen_frames": total_gen_frames,
        "videos": records,
    })
    return payload


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def _first16_complete(out_dir: Path, stem: str) -> bool:
    sub = out_dir / stem
    return all((sub / f"frame_{idx:02d}.png").exists() for idx in range(N_FVD_FRAMES))


def _stem_artifacts_complete(record: dict, *, pred_videos_dir: Path, pred_frames_dir: Path,
                             gt_frames_dir: Path, pred_first16_dir: Path,
                             gt_first16_dir: Path) -> bool:
    try:
        stem = record["stem"]
        n_gen = int(record["n_gen"])
        fvd_kept = int(record.get("fvd_kept_frames", 0))
    except Exception:
        return False

    if n_gen <= 0:
        return False
    if not (pred_videos_dir / f"{stem}.mp4").exists():
        return False
    for idx in range(n_gen):
        if not (pred_frames_dir / f"{stem}_f{idx:04d}.png").exists():
            return False
        if not (gt_frames_dir / f"{stem}_f{idx:04d}.png").exists():
            return False

    if n_gen >= N_FVD_FRAMES:
        if fvd_kept != N_FVD_FRAMES:
            return False
        return (_first16_complete(pred_first16_dir, stem)
                and _first16_complete(gt_first16_dir, stem))
    return fvd_kept == 0


def _all_records_artifacts_complete(records_by_stem: dict, stems: list, **artifact_dirs) -> bool:
    for stem in stems:
        record = records_by_stem.get(stem)
        if record is None:
            return False
        if not _stem_artifacts_complete(record, **artifact_dirs):
            return False
    return True


def _cleanup_stem_outputs(stem: str, *, pred_videos_dir: Path, pred_frames_dir: Path,
                          gt_frames_dir: Path, pred_first16_dir: Path,
                          gt_first16_dir: Path) -> None:
    mp4 = pred_videos_dir / f"{stem}.mp4"
    if mp4.exists():
        mp4.unlink()
    for out_dir in (pred_frames_dir, gt_frames_dir):
        for frame_path in out_dir.glob(f"{stem}_f*.png"):
            frame_path.unlink()
    for out_dir in (pred_first16_dir, gt_first16_dir):
        sub = out_dir / stem
        if sub.exists():
            shutil.rmtree(sub)


# ---------------------------------------------------------------------------
# Shard
# ---------------------------------------------------------------------------

def run_shard(cfg, step, rank, world_size, *, cfg_scale=3.0,
              fps=4, chunk_size=16, max_videos_per_shard=None,
              chunk_gen_frames=8, config_path=None):
    assert 0 <= rank < world_size, f"bad rank {rank} for world_size {world_size}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = f"chunkAuto{chunk_gen_frames}"
    log.info(f"(chunk auto) [shard {rank}/{world_size}] step={step} cfg_scale={cfg_scale} "
             f"(chunked autoregressive, GT re-anchor every {chunk_gen_frames} frames)")
    set_seed(cfg.train.seed + rank)

    # Dataloader is only used to get its dataset (we iterate dataset[idx] directly
    # to keep per-clip memory bounded; mirrors the worldgym shard). RT-1 has a single
    # relative-action dataloader (no absolute variant).
    dataloader = get_rt1_eval_dataloader(cfg, split="val", batch_size=1)
    dataset = dataloader.dataset
    n_total = len(dataset)
    my_indices = [i for i in range(n_total) if i % world_size == rank]
    if max_videos_per_shard is not None:
        my_indices = my_indices[:max_videos_per_shard]
    stems_all = [Path(p).stem for p in dataset.video_paths]
    my_stems = [stems_all[i] for i in my_indices]
    log.info(f"[shard {rank}/{world_size}] dataset size {n_total}, "
             f"this shard owns {len(my_indices)} clips")

    denoiser, actual_step = setup_model(cfg, step)
    denoiser = denoiser.to(device).eval()
    n_history = denoiser.num_history_steps
    log.info(f"[shard {rank}/{world_size}] loaded denoiser step={actual_step} "
             f"n_history={n_history}")

    log.info("Using SD3 VAE (encoding video latents on the fly)")
    sd3_vae = SD3VAE().to(device)

    # Output dirs (shared across all ranks).
    ckpt_tag = f"step{actual_step}"
    run_root = _build_run_root(cfg, ckpt_tag, cfg_scale, mode)
    pred_videos_dir = run_root / "pred_videos"
    pred_frames_dir = run_root / "pred_frames"
    gt_frames_dir = run_root / "gt_frames"
    pred_first16_dir = run_root / "pred_first16"
    gt_first16_dir = run_root / "gt_first16"
    shards_dir = run_root / "shards"
    for d in (pred_videos_dir, pred_frames_dir, gt_frames_dir,
              pred_first16_dir, gt_first16_dir, shards_dir):
        d.mkdir(parents=True, exist_ok=True)

    artifact_dirs = {
        "pred_videos_dir": pred_videos_dir,
        "pred_frames_dir": pred_frames_dir,
        "gt_frames_dir": gt_frames_dir,
        "pred_first16_dir": pred_first16_dir,
        "gt_first16_dir": gt_first16_dir,
    }
    out_json = shards_dir / f"metrics_rank{rank}_of{world_size}.json"
    done_flag = shards_dir / f"done_rank{rank}_of{world_size}.flag"
    metadata = _build_resume_metadata(
        config_path if config_path is not None else "",
        rank, world_size, actual_step,
        cfg_scale=cfg_scale, n_history=n_history,
        fps=fps, chunk_size=chunk_size,
        max_videos_per_shard=max_videos_per_shard,
        chunk_gen_frames=chunk_gen_frames,
    )

    records_by_stem, resume_msg = _load_resume_records(out_json, metadata, my_stems)
    log.info(f"[shard {rank}/{world_size}] resume: {resume_msg}")

    initial_complete = _all_records_artifacts_complete(records_by_stem, my_stems,
                                                       **artifact_dirs)
    if done_flag.exists() and not initial_complete:
        done_flag.unlink()
        log.info(f"[shard {rank}/{world_size}] removed stale done flag {done_flag}")
    if initial_complete:
        log.info(f"[shard {rank}/{world_size}] existing outputs are complete; "
                 "rerun will verify and skip")

    for shard_i, ds_idx in enumerate(my_indices):
        stem = stems_all[ds_idx]

        record = records_by_stem.get(stem)
        if record is not None and _stem_artifacts_complete(record, **artifact_dirs):
            log.info(f"[shard {rank}/{world_size}] ({shard_i+1}/{len(my_indices)}) "
                     f"{stem}: already complete, skipping")
            continue

        if record is not None:
            log.info(f"[shard {rank}/{world_size}] ({shard_i+1}/{len(my_indices)}) "
                     f"{stem}: incomplete artifacts, regenerating")
        records_by_stem.pop(stem, None)
        _cleanup_stem_outputs(stem, **artifact_dirs)

        mp4_path = pred_videos_dir / f"{stem}.mp4"
        sample = dataset[ds_idx]
        actions = sample['action'].unsqueeze(0).to(device)            # (1, L, action_dim)
        video_pix = sample['video'].unsqueeze(0).to(device)           # (1, L, H, W, C) [0,1]
        model_input = sd3_vae.encode(video_pix)

        L = int(model_input.shape[1])
        n_gen = L - n_history
        log.info(f"[shard {rank}/{world_size}] ({shard_i+1}/{len(my_indices)}) "
                 f"{stem} L={L} n_gen={n_gen}")
        if n_gen <= 0:
            log.warning(f"  skipping {stem}: L={L} <= n_history={n_history}")
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        with torch.no_grad():
            # sample_autoregressive does not mutate its obs input, so model_input
            # stays intact for the GT decode below.
            gen = denoiser.sample_autoregressive(
                model_input, actions,
                use_ema=True, init_val=None, cfg_scale=cfg_scale,
                gen_chunk=chunk_gen_frames,
            )  # (1, L, 3, H_pix, W_pix) in [-1, 1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_time = time.perf_counter() - _t0

        # GT pixels for metrics + frame dumps: VAE-decode the (unmutated) model input.
        with torch.no_grad():
            gt_pixels = decode_latents(model_input, chunk_size=chunk_size)  # (1, L, 3, H, W) [-1, 1]

        # Per-video metrics on (n_gen, 3, H, W) in [-1, 1].
        gen_g = gen[0, n_history:L].float().cpu()
        gt_g = gt_pixels[0, n_history:L].float().cpu()
        ssim_m = float(get_ssim(gen_g, gt_g).mean())
        psnr_m = float(get_psnr(gen_g, gt_g).mean())
        lpips_m = float(get_lpips(gen_g, gt_g).mean())

        log.info(f"  (chunk auto) metrics: SSIM={ssim_m:.4f} "
                 f"PSNR={psnr_m:.4f} LPIPS={lpips_m:.4f} | gen_time={gen_time:.2f}s")

        # Save predicted MP4 (full clip, history+generated, in [-1, 1]).
        save_video(gen[0].cpu(), str(mp4_path), fps=fps, value_range=(-1, 1))

        # Dump GENERATED frames (exclude the n_history GT seed frames, so FID
        # measures actual model output).
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

        records_by_stem[stem] = {
            "stem": stem,
            "n_gen": n_gen,
            "ssim": ssim_m,
            "psnr": psnr_m,
            "lpips": lpips_m,
            "gen_time_s": gen_time,
            "fvd_kept_frames": fvd_kept,
        }

        complete = (
            len(records_by_stem) >= len(my_stems)
            and _all_records_artifacts_complete(records_by_stem, my_stems, **artifact_dirs)
        )
        _write_json_atomic(
            out_json,
            _build_shard_payload(
                metadata, records_by_stem, my_stems,
                config_path=config_path, step=step,
                total_available_episodes=n_total, complete=complete,
            ),
        )
        log.info(f"[shard {rank}/{world_size}] saved progress to {out_json} "
                 f"({len(records_by_stem)}/{len(my_stems)} records)")
        log.info(f"[shard {rank}/{world_size}] (chunk auto) running avg: "
                 f"{_running_summary(_ordered_records(records_by_stem, my_stems))}")

    final_complete = _all_records_artifacts_complete(records_by_stem, my_stems,
                                                     **artifact_dirs)
    records = _ordered_records(records_by_stem, my_stems)
    total_gen_time, total_gen_frames = _record_totals(records)
    final_tpf = (total_gen_time / total_gen_frames) if total_gen_frames > 0 else float('nan')
    status = "complete" if final_complete else "incomplete"
    log.info(f"[shard {rank}/{world_size}] {status}. {len(records)} clips, "
             f"gen_time={total_gen_time:.1f}s frames={total_gen_frames} "
             f"time/frame={final_tpf:.4f}s")
    log.info(f"[shard {rank}/{world_size}] final running avg: {_running_summary(records)}")

    _write_json_atomic(
        out_json,
        _build_shard_payload(
            metadata, records_by_stem, my_stems,
            config_path=config_path, step=step,
            total_available_episodes=n_total, complete=final_complete,
        ),
    )
    if final_complete:
        done_flag.touch()
        log.info(f"[shard {rank}/{world_size}] wrote {out_json} and touched {done_flag}")
    else:
        if done_flag.exists():
            done_flag.unlink()
        missing = []
        for stem in my_stems:
            record = records_by_stem.get(stem)
            if record is None or not _stem_artifacts_complete(record, **artifact_dirs):
                missing.append(stem)
        log.warning(f"[shard {rank}/{world_size}] wrote incomplete progress to {out_json}; "
                    f"not touching done flag. Missing/incomplete stems: {missing[:10]}"
                    f"{' ...' if len(missing) > 10 else ''}")


def _build_parser():
    p = argparse.ArgumentParser(description="CFG denoiser multi-GPU eval -- per-shard")
    p.add_argument("--config-path", required=True, help="Hydra/OmegaConf YAML")
    p.add_argument("--step", type=int, default=None,
                   help="Checkpoint step to load (default: latest)")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=16,
                   help="GT VAE decode chunk size")
    p.add_argument("--max-videos-per-shard", type=int, default=None,
                   help="Optional cap for smoke tests")
    p.add_argument("--chunk-gen-frames", type=int, default=8,
                   help="Frames generated per chunk before re-anchoring history on ground truth "
                        "(default 8).")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    cfg = OmegaConf.load(args.config_path)
    run_shard(
        cfg, args.step, args.rank, args.world_size,
        cfg_scale=args.cfg_scale,
        fps=args.fps,
        chunk_size=args.chunk_size,
        max_videos_per_shard=args.max_videos_per_shard,
        chunk_gen_frames=args.chunk_gen_frames,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
