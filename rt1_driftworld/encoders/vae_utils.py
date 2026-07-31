"""
Utilities for decoding SD3 VAE latents back to pixels
"""
import torch
from diffusers import AutoencoderKL

def decode_latents(latents, vae=None, chunk_size=16):
    """
    Decode video latents to pixels with the SD3 VAE, under torch.no_grad().
    Args:
        latents: (B, T, 16, H/8, W/8) video latents
        vae: pre-loaded SD3 AutoencoderKL. If None, it is loaded.
        chunk_size: frames decoded at once, to bound peak memory
    Returns:
        (B, T, 3, H, W) pixels in [-1, 1]
    """
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            subfolder="vae",
            torch_dtype=latents.dtype,
        ).to(latents.device)
        vae.eval()

    B, T, C, H_latent, W_latent = latents.shape

    flat_latents = latents.reshape(B * T, C, H_latent, W_latent)
    flat_latents = flat_latents / vae.config.scaling_factor # Undo the scaling applied at encode time

    decoded_list = []
    with torch.no_grad():
        for start_idx in range(0, flat_latents.size(0), chunk_size):
            chunk = flat_latents[start_idx : start_idx + chunk_size]
            decoded_chunk = vae.decode(chunk, return_dict=False)[0]
            decoded_list.append(decoded_chunk)

    decoded = torch.cat(decoded_list, dim=0)

    _, C_out, H, W = decoded.shape
    return decoded.reshape(B, T, C_out, H, W)

def decode_latents_with_grad(latents, vae, chunk_size=16):
    """
    Like `decode_latents`, but without `torch.no_grad()`, so gradients flow
    back through the VAE decoder into the latents. Used when running a pixel-space
    feature extractor (e.g. DINOv3) on the generator's latent-space output.
    Args:
        latents: (B, T, 16, H/8, W/8) video latents
        vae: pre-loaded SD3 AutoencoderKL
        chunk_size: frames decoded at once, to bound peak activation memory
    Returns:
        (B, T, 3, H, W) pixels in [-1, 1]
    """
    assert vae is not None, "decode_latents_with_grad requires a caller-provided VAE"
    B, T, C, H_latent, W_latent = latents.shape

    flat_latents = latents.reshape(B * T, C, H_latent, W_latent)
    flat_latents = flat_latents / vae.config.scaling_factor

    decoded_list = []
    for start_idx in range(0, flat_latents.size(0), chunk_size):
        chunk = flat_latents[start_idx : start_idx + chunk_size]
        decoded_chunk = vae.decode(chunk, return_dict=False)[0]
        decoded_list.append(decoded_chunk)

    decoded = torch.cat(decoded_list, dim=0)
    _, C_out, H, W = decoded.shape
    return decoded.reshape(B, T, C_out, H, W)
