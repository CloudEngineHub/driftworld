"""
Training loop for DriftWorld on RT-1 -- phase-2 self-forcing
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import contextlib
from collections import defaultdict
import torch
torch.set_float32_matmul_precision('high')
import numpy as np
import wandb
import logging
import random
import json

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.rt1_dataloader import get_rt1_dataloader
from utils.model_utils import create_model_selfforce
from encoders.vae_sd3 import VAE

log = logging.getLogger(__name__)


def ddp_setup():
    """
    Initialize torch.distributed from env vars set by torchrun.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def is_main():
    return int(os.environ.get("RANK", 0)) == 0


def barrier(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.barrier()


def set_seed(seed):
    if is_main():
        log.info(f"Seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


def phase2_perstep_step(denoiser, inner, nbatch, device, world_size, cfg, optimizer, scheduler):
    """
    One optimizer step of phase-2 self-forcing, where backward is applied per-step.
    """
    prep = inner.phase2_prepare(nbatch, device)
    obs = prep['obs']; rollout = prep['rollout']; act_k = prep['act_k']
    cfg_scale_k = prep['cfg_scale_k']; uncond_w_b = prep['uncond_w_b']; do_uncond = prep['do_uncond']
    b = prep['b']; n = prep['n']; c = prep['c']; h = prep['h']; w = prep['w']
    k = prep['k']; seq_length = prep['seq_length']

    loss_val = 0.0
    metrics = defaultdict(float)

    for i in range(seq_length):
        log.info(f"(iter {i}/{seq_length})")
        # .clone() keeps this history independent of the in-place rollout write below
        prev_obs = rollout[:, i:n + i].reshape(k * b, n * c, h, w).clone()
        prev_act = act_k[:, i:n + i]
        target_obs = obs[:, n + i].unsqueeze(1)
        # Action accentuation: preceding real frame GT_t as the no-action negative
        uncond_obs = obs[:, n + i - 1].unsqueeze(1) if do_uncond else None
        # Past frame o_t for token-space motion weighting
        past_obs = obs[:, n + i - 1].unsqueeze(1) if inner.motion_masking else None
        noise = torch.randn((k * b, c, h, w), device=device)

        # Suppress DDP's own all-reduce on every step; we average gradients manually after the rollout
        sync_ctx = denoiser.no_sync() if world_size > 1 else contextlib.nullcontext()
        with sync_ctx:
            loss_i, gen_det, info_i = denoiser(
                prev_obs, prev_act, cfg_scale_k, noise,
                target_obs, uncond_obs, uncond_w_b, past_obs, b, k)
            (loss_i / seq_length).backward()

        # detach so the next iteration's history carries no graph
        rollout[:, n + i] = gen_det
        loss_val += (loss_i.detach() / seq_length).item()
        for key, value in info_i.items():
            metrics[key] += value

    # Manual gradient averaging across ranks
    if world_size > 1:
        for p in inner.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)

    torch.nn.utils.clip_grad_norm_(inner.parameters(), max_norm=cfg.opt.grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    inner.update_ema()

    averages = {key: total / seq_length for key, total in metrics.items()}
    averages['loss_backprop'] = loss_val
    return averages


def train(cfg):
    """
    Train model (phase-2 self-forcing only), given hydra config cfg. Multi-GPU via torchrun + DDP.
    """
    if not cfg.model.is_phase_2:
        raise ValueError(
            "train_selfforce.py runs phase-2 self-forcing only; set model.is_phase_2=True in the config."
        )

    local_rank, rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main():
        log.info(f"Using device: {device} | world_size={world_size}")

    set_seed(cfg.train.seed)

    if is_main():
        log.info("Creating dataloader for RT-1")
    dataloader = get_rt1_dataloader(
        cfg, split="train",
        rank=rank, world_size=world_size,
    )

    if is_main():
        log.info("Creating model (self-forcing variant)")
    denoiser = create_model_selfforce(cfg, device)

    if is_main():
        log.info("Forward pass: Phase 2 self-forcing")
    denoiser.forward = denoiser.phase2_step

    if world_size > 1:
        denoiser = DDP(
            denoiser,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        inner = denoiser.module
    else:
        inner = denoiser

    if is_main():
        log.info("Creating optimizer")
    optimizer = torch.optim.AdamW(
        params=inner.parameters(),
        lr=cfg.opt.lr,
        betas=(0.9, cfg.opt.beta2),
        weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6 / cfg.opt.lr,
        end_factor=1.0,
        total_iters=cfg.opt.warmup_steps
    )

    actual_step = 0 # current step
    to_skip = False # Whether to skip to the correct location in dataloader

    if is_main():
        log.info("Creating model / Restoring checkpoint")
        os.makedirs(f"{cfg.output_dir}/ckpt_save", exist_ok=True)
    barrier(world_size)

    # Load checkpoint
    if os.path.exists(cfg.path_ckpt_latest):
        ckpt = torch.load(cfg.path_ckpt_latest, map_location="cpu", weights_only=False)
        inner.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        actual_step = ckpt['step'] + 1
        del ckpt
        if is_main():
            log.info(f"Restored from step {actual_step} ckpt")
        if actual_step % len(dataloader) != 0:
            to_skip = True
    elif os.path.exists(cfg.path_ckpt_phase1):
        # Just started phase 2, so load from phase 1 checkpoint
        ckpt = torch.load(cfg.path_ckpt_phase1, map_location="cpu", weights_only=False)
        inner.load_state_dict(ckpt['model'])
        if is_main():
            log.info(f"Restored from phase 1's step {ckpt['step']} ckpt to begin phase 2 training")
        del ckpt
    barrier(world_size)

    # Set up wandb run
    if is_main():
        wandb.login(key=cfg.wandb_info.key)
        if not os.path.exists(cfg.wandb_info.saved_run_id):
            run = wandb.init(
                entity=cfg.wandb_info.entity,
                project=cfg.wandb_info.project,
                name=cfg.wandb_info.name,
                dir=cfg.output_dir
            )
            run_id = run.id
            with open(cfg.wandb_info.saved_run_id, 'w') as f:
                json.dump({'run_id': run_id}, f)
            log.info(f"Started new wandb run {run_id}")
        else:
            log.info(f"Resuming wandb run")
            with open(cfg.wandb_info.saved_run_id, 'r') as f:
                run_id = json.load(f)['run_id']
            run = wandb.init(
                entity=cfg.wandb_info.entity,
                project=cfg.wandb_info.project,
                id=run_id,
                resume="allow",
                dir=cfg.output_dir
            )

        n_params = sum(p.numel() for p in inner.parameters() if p.requires_grad)
        wandb.log({'num_params': n_params, 'seed': cfg.train.seed}, step=0)
    barrier(world_size)

    if is_main():
        log.info("Using SD3 VAE")
    vae = VAE().to(device)
    if is_main():
        log.info("Using SD3 VAE: done loading")

    start_ep = actual_step // len(dataloader) + 1
    for epoch_idx in range(start_ep, cfg.train.num_epochs + 1):
        if world_size > 1 and hasattr(dataloader, "sampler") and isinstance(
                dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            dataloader.sampler.set_epoch(epoch_idx)

        if is_main():
            log.info(f"(epoch {epoch_idx}) start")
        for batch_idx, nbatch in enumerate(dataloader):
            # Reach actual_step by skipping forward in dataloader
            if to_skip:
                cur_step = len(dataloader) * (epoch_idx - 1) + batch_idx
                if cur_step < actual_step:
                    continue
                elif cur_step == actual_step:
                    to_skip = False # reached the correct location, no need to skip further

            # Compute VAE latent
            nbatch['video_latents'] = vae.encode(nbatch['video'].to(device))

            if batch_idx == 0 and epoch_idx == 1 and is_main():
                log.info(f"action: {nbatch['action'].shape} | min {nbatch['action'].min():.4f} | max {nbatch['action'].max():.4f}")
                log.info(f"video_latents: {nbatch['video_latents'].shape} | min {nbatch['video_latents'].min():.4f} | max {nbatch['video_latents'].max():.4f}")

            # Phase-2 self-forcing with per-step backward (backward + step inside)
            metrics = phase2_perstep_step(
                denoiser, inner, nbatch, device, world_size, cfg, optimizer, scheduler)

            metrics['lr'] = scheduler.get_last_lr()[0]
            if is_main():
                wandb.log(metrics, step=actual_step)
                log.info(f"(epoch {epoch_idx}) (batch {batch_idx}/{len(dataloader)})")
                for k, v in metrics.items():
                    if k.startswith("loss_") or k == "lr":
                        log.info(f"{k}: {v}")

            # Save checkpoint
            if actual_step % cfg.train.ckpt_every == 0:
                if is_main():
                    log.info(f"(epoch {epoch_idx}) Saving latest ckpt at step {actual_step}")
                    if os.path.exists(cfg.path_ckpt_latest):
                        try:
                            os.replace(cfg.path_ckpt_latest, cfg.path_ckpt_2nd_latest)
                        except Exception as e:
                            log.info(f"Failed to move latest checkpoint to 2nd latest: {e}")
                    checkpoint = {
                        'step': actual_step,
                        'model': inner.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }
                    torch.save(checkpoint, cfg.path_ckpt_latest)
                barrier(world_size)

            actual_step += 1

    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
