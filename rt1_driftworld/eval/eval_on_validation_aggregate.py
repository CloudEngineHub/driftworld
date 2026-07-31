"""
CFG denoiser multi-GPU eval -- POST-HOC aggregation + FID + FVD.

Run this AFTER all `world_size` ranks of eval_on_validation_shard.py
have finished and written their shards/done_rank{N}_of{M}.flag.

    python -m eval.eval_on_validation_aggregate \
        --config-path <same hydra cfg the shards used> \
        --step 100000 --world-size 8 --cfg-scale 1.5

Pass the same --chunk-gen-frames the shards used so the run_root resolves correctly.

The script:
  1. Resolves the same run_root the shards wrote to.
  2. Verifies all `world_size` shards completed (per-rank flag + JSON) and
     used the same actual_step.
  3. Aggregates per-video SSIM/PSNR/LPIPS into means + stds.
  4. Runs pytorch-fid as a subprocess over pred_frames/ vs gt_frames/.
     Path: cfg.eval.pytorch_fid_dir/src/pytorch_fid/fid_score.py
  5. Runs stylegan-v calc_metrics_for_dataset.py with --metrics fvd2048_16f
     over pred_first16/ vs gt_first16/.
     Path: cfg.eval.stylegan_v_dir/src/scripts/calc_metrics_for_dataset.py
  6. Writes aggregate_summary.json under the run_root.
"""
import re
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .eval_on_validation_shard import _build_run_root

log = logging.getLogger(__name__)


def _check_all_shards_done(shards_dir: Path, world_size: int):
    missing_flags, missing_jsons = [], []
    for r in range(world_size):
        if not (shards_dir / f"done_rank{r}_of{world_size}.flag").exists():
            missing_flags.append(r)
        if not (shards_dir / f"metrics_rank{r}_of{world_size}.json").exists():
            missing_jsons.append(r)
    if missing_flags or missing_jsons:
        raise RuntimeError(
            f"shards not all done yet under {shards_dir}: "
            f"missing flags for ranks {missing_flags}, "
            f"missing JSONs for ranks {missing_jsons}"
        )


def _aggregate_per_video(shards_dir: Path, world_size: int):
    all_records = []
    actual_steps = set()
    total_gen_time = 0.0
    total_gen_frames = 0
    for r in range(world_size):
        with open(shards_dir / f"metrics_rank{r}_of{world_size}.json") as f:
            shard = json.load(f)
        all_records.extend(shard["videos"])
        if "actual_step" in shard:
            actual_steps.add(shard["actual_step"])
        total_gen_time += float(shard.get("total_gen_time_s", 0.0))
        total_gen_frames += int(shard.get("total_gen_frames", 0))

    if len(actual_steps) > 1:
        raise RuntimeError(
            f"shards used different actual_step values: {sorted(actual_steps)}"
        )
    actual_step = next(iter(actual_steps)) if actual_steps else None

    stems = [r["stem"] for r in all_records]
    if len(stems) != len(set(stems)):
        dupes = {s for s in stems if stems.count(s) > 1}
        raise RuntimeError(f"duplicate stems across shards: {sorted(dupes)[:10]} ...")

    def _meanstd(key):
        vals = np.array([r[key] for r in all_records], dtype=np.float64)
        return float(vals.mean()), float(vals.std())

    summary = {"n_videos": len(all_records), "actual_step": actual_step}
    for k in ("ssim", "psnr", "lpips"):
        mean, std = _meanstd(k)
        summary[k] = mean
        summary[f"{k}_std"] = std
    summary["total_gen_time_s"] = total_gen_time
    summary["total_gen_frames"] = total_gen_frames
    summary["time_per_frame_s"] = (
        total_gen_time / total_gen_frames if total_gen_frames > 0 else float('nan')
    )
    return summary, all_records


def _run_pytorch_fid(cfg, gt_frames_dir: Path, pred_frames_dir: Path) -> float:
    fid_repo = Path(cfg.eval.pytorch_fid_dir)
    fid_script = fid_repo / "src" / "pytorch_fid" / "fid_score.py"
    if not fid_script.exists():
        raise FileNotFoundError(
            f"pytorch-fid script not found at {fid_script}. "
            f"Set cfg.eval.pytorch_fid_dir to the pytorch-fid repo root."
        )
    cmd = [sys.executable, str(fid_script), str(gt_frames_dir), str(pred_frames_dir)]
    log.info(f"running pytorch-fid: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(fid_repo),
                            capture_output=True, text=True, check=True)
    log.info(f"pytorch-fid stdout:\n{result.stdout}")
    m = re.search(r"FID:\s*([-+]?\d+(?:\.\d+)?)", result.stdout)
    if not m:
        raise RuntimeError(f"could not parse FID from pytorch-fid stdout: {result.stdout!r}")
    return float(m.group(1))


def _run_stylegan_v_fvd(cfg, gt_first16_dir: Path, pred_first16_dir: Path,
                        metric: str = "fvd2048_16f", resolution: int = 256) -> float:
    sv_repo = Path(cfg.eval.stylegan_v_dir)
    sv_script = sv_repo / "src" / "scripts" / "calc_metrics_for_dataset.py"
    if not sv_script.exists():
        raise FileNotFoundError(
            f"stylegan-v script not found at {sv_script}. "
            f"Set cfg.eval.stylegan_v_dir to the stylegan-v repo root."
        )
    cmd = [
        sys.executable, str(sv_script),
        "--real_data_path", str(gt_first16_dir),
        "--fake_data_path", str(pred_first16_dir),
        "--mirror", "1",
        "--gpus", "1",
        "--resolution", str(resolution),
        "--metrics", metric,
        "--verbose", "0",
        "--use_cache", "0",
    ]
    log.info(f"running stylegan-v: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(sv_repo),
                            capture_output=True, text=True, check=True)
    log.info(f"stylegan-v stdout:\n{result.stdout}")
    m = re.search(r'\{"results":.*?\}', result.stdout)
    if not m:
        raise RuntimeError(f"could not parse FVD from stylegan-v stdout: {result.stdout!r}")
    obj = json.loads(m.group() + "}")
    return float(obj["results"][metric])


def aggregate(cfg, step, world_size, *, cfg_scale=3.0,
              skip_fid=False, skip_fvd=False, fvd_metric="fvd2048_16f",
              chunk_gen_frames=8):
    mode = f"chunkAuto{chunk_gen_frames}"
    ckpt_tag = f"step{step}" if step is not None else "stepLatest"
    run_root = _build_run_root(cfg, ckpt_tag, cfg_scale, mode)
    log.info(f"aggregating under {run_root}")

    shards_dir = run_root / "shards"
    _check_all_shards_done(shards_dir, world_size)

    summary, records = _aggregate_per_video(shards_dir, world_size)
    log.info(
        f"[summary] per-video averages over {summary['n_videos']} videos "
        f"(actual_step={summary['actual_step']}): "
        f"SSIM={summary['ssim']:.4f} "
        f"PSNR={summary['psnr']:.4f} LPIPS={summary['lpips']:.4f}"
    )
    log.info(
        f"[summary] generation throughput: total gen_time={summary['total_gen_time_s']:.1f}s "
        f"over {summary['total_gen_frames']} frames | "
        f"time/frame={summary['time_per_frame_s']:.4f}s"
    )

    if not skip_fid:
        fid = _run_pytorch_fid(cfg,
                               run_root / "gt_frames",
                               run_root / "pred_frames")
        summary["fid"] = fid
        log.info(f"[summary] FID = {fid:.4f}")
    else:
        log.info("skipping FID (--skip-fid)")

    if not skip_fvd:
        fvd = _run_stylegan_v_fvd(cfg,
                                  run_root / "gt_first16",
                                  run_root / "pred_first16",
                                  metric=fvd_metric)
        summary[fvd_metric] = fvd
        log.info(f"[summary] {fvd_metric} = {fvd:.4f}")
    else:
        log.info("skipping FVD (--skip-fvd)")

    summary.update({
        "world_size": world_size,
        "step": step,
        "cfg_scale": cfg_scale,
        "chunk_gen_frames": chunk_gen_frames,
    })
    out = run_root / "aggregate_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"wrote {out}")
    return summary


def _build_parser():
    p = argparse.ArgumentParser(description="CFG denoiser multi-GPU eval -- aggregate + FID + FVD")
    p.add_argument("--config-path", required=True)
    p.add_argument("--step", type=int, default=None,
                   help="Checkpoint step the shards used (default: latest -> stepLatest tag)")
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--cfg-scale", type=float, default=3.0)
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--skip-fvd", action="store_true")
    p.add_argument("--fvd-metric", default="fvd2048_16f")
    p.add_argument("--chunk-gen-frames", type=int, default=8,
                   help="Frames per chunk; must match the shards' --chunk-gen-frames so the "
                        "run_root resolves to the directory they wrote to (default 8).")
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    cfg = OmegaConf.load(args.config_path)
    aggregate(
        cfg, args.step, args.world_size,
        cfg_scale=args.cfg_scale,
        skip_fid=args.skip_fid,
        skip_fvd=args.skip_fvd,
        fvd_metric=args.fvd_metric,
        chunk_gen_frames=args.chunk_gen_frames,
    )


if __name__ == "__main__":
    main()
