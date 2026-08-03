"""
DriftWorld denoiser for Language Table
"""
import torch
import torch.nn as nn
import logging
from collections import defaultdict
import copy
from einops import rearrange

from drifting.drift_loss_indep import drift_loss
from unet.unet import InnerModel
from unet.unet_configs import get_inner_model_config

log = logging.getLogger(__name__)

class Denoiser(nn.Module):
    LATENT_KEY = "_latent"

    def __init__(self, cfg):
        super().__init__()
        inner_model_cfg = get_inner_model_config(cfg.model.inner_model_cfg)
        assert inner_model_cfg.num_actions == cfg.language_table.action_dim, "Action dim of U-Net does not match action dim of dataset"
        self.model = InnerModel(inner_model_cfg)

        self.temp_list = tuple(list(cfg.model.temp_list))
        self.n_neg = cfg.model.n_neg
        self.num_history_steps = inner_model_cfg.num_steps_conditioning

        self.decay = cfg.train.decay

        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)

        assert cfg.model.vae_type == "sd3", f"only the SD3 VAE is supported, got {cfg.model.vae_type!r}"
        self.chunk_size = cfg.model.chunk_size

        ##############################################################################
        ### Drifting loss in latent space
        self.latent_loss_type = getattr(cfg.model, 'latent_loss_type', None)
        assert self.latent_loss_type in (None, "latent_spatial"), f"got unrecognized latent_loss_type {self.latent_loss_type!r}"

        ##############################################################################
        ### Token-space motion weighting
        if getattr(cfg, 'mask', None) is not None:
            from features.motion_mask import TokenMasker
            mk = cfg.mask
            self.masker = TokenMasker(
                motion_mask_alpha=mk.get("motion_mask_alpha", 2.5),
                motion_mask_lambda=mk.get("motion_mask_lambda", None),
                motion_mask_qhi=mk.get("motion_mask_qhi", 0.98),
                motion_mask_tau=mk.get("motion_mask_tau", 0.35),
            )
        else:
            self.masker = None
        self.motion_masking = self.masker is not None and self.masker.motion_masking_enabled

    def _feature_activations(self, latents: torch.Tensor):
        """
        Create the drift fields for a batch of latent frames.
        Args:
            latents: (B, M, C, H, W) latent frames
        Returns:
            dict[str, (B*M, T, D)] of activations (one entry per aggregation) where T = number of
            drift tokens (spatial latent cells / patches) per individual frame.
        """
        B, M, C, H, W = latents.shape

        out = {}
        if self.latent_loss_type == "latent_spatial":
            # one drift field per spatial latent cell: H*W tokens of dimension C
            out[self.LATENT_KEY] = latents.reshape(B * M, C, H * W).transpose(1, 2)
        return out

    def _drift_from_act(self, gen_act: dict, pos_act: dict, B: int, token_weights: dict = None):
        """
        Apply the drifting loss over a dict of feature activations.
        Args:
            gen_act: dict of (B*N, T, D) generated activations
            pos_act: dict of (B, T, D) positive (data) activations
            B: batch size
            token_weights: optional dict of (B, T) per-token loss multipliers (token-space
                motion masking). Applied per key; keys absent from it are left unweighted.
        Returns:
            (scalar loss, info dict)
        """
        loss = 0.0
        info = {}
        for key_name in gen_act.keys():
            _, T, D = pos_act[key_name].shape
            p_act = pos_act[key_name].detach().reshape(B*T, 1, D)
            g_act = rearrange(gen_act[key_name], '(b n) t d -> (b t) n d', b=B)

            layer_loss, layer_info = drift_loss(gen=g_act, fixed_pos=p_act,
                                                R_list=self.temp_list)

            if token_weights is not None and key_name in token_weights:
                w_flat = token_weights[key_name].reshape(-1).to(layer_loss.dtype)
                loss = loss + (layer_loss * w_flat).mean()
            else:
                loss = loss + layer_loss.mean()
            for k, v in layer_info.items():
                info[k] = info.get(k, 0.0) + v
        return loss, info

    def drifting_loss(self, gen: torch.Tensor, pos: torch.Tensor, past: torch.Tensor = None):
        """
        Drifting loss: MSE(gen, stopgrad(gen + V)).
        Args:
            gen: Generated samples [B, N, C, H, W]
            pos: Data samples [B, 1, C, H, W]
            past: optional ground-truth past frame o_t [B, 1, C, H, W]
        Returns:
            scalar drifting loss (which is the mean over all entries in the batch) and dict of metrics
        """
        B, N, C, H, W = gen.shape
        if self.latent_loss_type is None:
            gen_flat = gen.reshape(B, N, -1) # [B, N, D], where D=C*H*W
            pos_flat = pos.reshape(B, 1, -1) # [B, 1, D]
            loss, info = drift_loss(gen=gen_flat, fixed_pos=pos_flat, R_list=self.temp_list)
            return loss.mean(), info

        # gen_act is a dictionary of (B*N, T, D) tensors
        # pos_act is a dictionary of (B, T, D) tensors
        gen_act = self._feature_activations(gen)
        pos_act = self._feature_activations(pos)

        # Token-space motion weighting
        token_weights = None
        if self.motion_masking and past is not None:
            with torch.no_grad():
                past_act = self._feature_activations(past)
            token_weights = self.masker.motion_token_weights(pos_act, past_act)

        loss, info = self._drift_from_act(gen_act, pos_act, B, token_weights=token_weights)
        return loss, info

    def forward(self, batch, device):
        """
        Forward pass to train DriftWorld
        Args:
            batch: dictionary representing a batch of data:
                - video_latents: (B, F, C, H, W) tensor of video latents
                - action: (B, F, action_dim) tensor of actions
            device: device
        Returns:
            Loss
        """
        obs = batch['video_latents'].to(device)
        act = batch['action'].to(device)

        b, t, c, h, w = obs.size()
        n = self.num_history_steps # num of history frames the U-Net conditions on
        seq_length = t - n # num future frames the model needs to predict in this sequence

        loss = 0
        metrics = defaultdict(float)

        # autoregressive rollout:
        #   i=0: frames 0..n-1 -> predict timestep n
        #   i=1: frames 1..n -> predict timestep n+1
        #   i=t-n-1: frames t-n-1..t-2 -> predict timestep t-1
        for i in range(seq_length):
            log.info(f"(iter {i}/{seq_length})")
            prev_obs = obs[:, i:n + i].reshape(b, n * c, h, w)
            prev_act = act[:, i:n + i]
            target_obs = obs[:, n + i].unsqueeze(1) # ground-truth target frame to predict

            # Past frame o_t (the newest history frame) for token-space motion masking (ground truth).
            past_obs = obs[:, n + i - 1].unsqueeze(1) if self.motion_masking else None

            prev_obs = prev_obs.repeat_interleave(self.n_neg, dim=0)
            prev_act = prev_act.repeat_interleave(self.n_neg, dim=0)

            # k = self.n_neg
            # prev_obs: (k*b, n*c, h, w)
            # prev_act: (k*b, n, 2)
            # target_obs: (b, 1, c, h, w), which is the single positive sample
            noise = torch.randn((self.n_neg * b, c, h, w), device=device)

            gen = self.model(noise, prev_obs, prev_act)
            gen = gen.reshape((b, self.n_neg, c, h, w))

            loss_i, info_i = self.drifting_loss(gen, target_obs, past=past_obs)
            loss += loss_i

            for key, value in info_i.items():
                metrics[key] += value

        loss /= seq_length
        averages = {key: total / seq_length for key, total in metrics.items()}
        averages['loss_backprop'] = loss.item()
        return loss, averages

    @torch.no_grad()
    def sample(self, prev_obs, prev_act, use_ema=True, init_val=None):
        """
        Sample from DriftWorld.
        Args:
            prev_obs: (B, N, C, H, W) history observations
            prev_act: (B, N, action_dim) history actions
            use_ema: whether to use EMA model
            init_val: initial value
        Returns:
            (B, C, H, W) prediction of the next frame
        """
        B, N, C, H, W = prev_obs.size()
        if init_val is None:
            init_tensor = torch.randn((B, C, H, W), device=prev_obs.device)
        else:
            init_tensor = torch.full((B, C, H, W), init_val, device=prev_obs.device)

        obs_flat = prev_obs.reshape((B, N*C, H, W))

        model = self.model if not use_ema else self.ema_model
        return model(init_tensor, obs_flat, prev_act)

    @torch.no_grad()
    def sample_autoregressive(self, obs, act, use_ema=True, init_val=None):
        """
        Sample from DriftWorld autoregressively.
        Args:
            obs: (B, T, C, H, W) frames. This provides the history frames, and the model will predict future frames.
            act: (B, T, action_dim) all actions
            use_ema: whether to use EMA
        Returns:
            (B, T, 3, H_out, W_out) tensor containing ground-truth history + predicted future frames in pixel space
        """
        b, t, c, h, w = obs.size()
        n = self.num_history_steps
        seq_length = t - n

        for i in range(seq_length):
            log.info(f"(iter {i}/{seq_length})")
            prev_obs = obs[:, i:n + i]
            prev_act = act[:, i:n + i]

            gen = self.sample(prev_obs, prev_act, use_ema, init_val)
            obs[:, n + i] = gen

        from encoders.vae_utils import decode_latents
        return decode_latents(obs, chunk_size=self.chunk_size)

    @torch.no_grad()
    def sample_autoregressive_act_len(self, obs, act, use_ema=True, init_val=None):
        """
        Sample from DriftWorld autoregressively.
        Args:
            obs: (B, T_obs, C, H, W) frames. This provides the history frames, and the model will predict future frames.
            act: (B, T, action_dim) all actions
            use_ema: whether to use EMA
        Returns:
            (B, T, 3, H_out, W_out) tensor containing ground-truth history + predicted future frames in pixel space
        """
        b, _, c, h, w = obs.size()
        t = act.shape[1]
        n = self.num_history_steps
        seq_length = t - n

        result = torch.zeros((b, t, c, h, w), device=obs.device)
        result[:, :n] = obs[:, :n]

        for i in range(seq_length):
            log.info(f"(iter {i}/{seq_length})")
            prev_obs = result[:, i:n + i]
            prev_act = act[:, i:n + i]
            gen = self.sample(prev_obs, prev_act, use_ema, init_val)
            result[:, n + i] = gen

        from encoders.vae_utils import decode_latents
        return decode_latents(result, chunk_size=self.chunk_size)

    @torch.no_grad()
    def sample_not_autoregressive(self, obs, act, use_ema=True, init_val=None):
        """
        Sample from DriftWorld non-autoregressively.
        Args:
            obs: (B, T_obs, C, H, W) frames. This provides the history frames, and the model will predict future frames.
            act: (B, T, action_dim) all actions
            use_ema: whether to use EMA
        Returns:
            (B, T, 3, H_out, W_out) tensor containing ground-truth history + predicted future frames IN PIXEL-SPACE
        """
        assert obs.shape[1] == act.shape[1], f"sample_not_autoregressive is called with obs {obs.shape} and act {act.shape} containing different numbers of frames"
        b, _, c, h, w = obs.size()
        t = act.shape[1]
        n = self.num_history_steps
        seq_length = t - n

        result = torch.zeros((b, t, c, h, w), device=obs.device)
        result[:, :n] = obs[:, :n]

        for i in range(seq_length):
            log.info(f"(iter {i}/{seq_length})")
            prev_obs = obs[:, i:n + i]
            prev_act = act[:, i:n + i]

            gen = self.sample(prev_obs, prev_act, use_ema, init_val)
            result[:, n + i] = gen

        from encoders.vae_utils import decode_latents
        return decode_latents(result, chunk_size=self.chunk_size)

    @torch.no_grad()
    def update_ema(self):
        """
        Updates the EMA parameters.
        """
        for p_ema, p_net in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.lerp_(p_net, 1 - self.decay)
