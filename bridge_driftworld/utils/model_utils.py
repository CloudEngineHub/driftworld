import logging

log = logging.getLogger(__name__)

LOSS_SIDE_PREFIXES = ("dinov3.", "feat_vae.")

def load_denoiser_state_dict(denoiser, state_dict):
    """
    Load a Denoiser state dict. Accepts checkpoints that have had the frozen loss-side modules stripped out.
    """
    missing, unexpected = denoiser.load_state_dict(state_dict, strict=False)
    bad_missing = [k for k in missing if not k.startswith(LOSS_SIDE_PREFIXES)]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch: missing={bad_missing}, unexpected={list(unexpected)}")
    if missing:
        log.info(f"Loaded a stripped checkpoint: {len(missing)} loss-side keys "
                 f"({', '.join(LOSS_SIDE_PREFIXES)}) kept at their newly initialized values")

def create_model(cfg, device):
    from drifting_denoiser import Denoiser
    return Denoiser(cfg).to(device)
