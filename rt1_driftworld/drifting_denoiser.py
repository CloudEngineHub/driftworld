"""
DriftWorld denoiser for RT-1
"""
import math
import torch
import torch.nn as nn
import logging
from collections import defaultdict
import copy
from einops import rearrange

from drifting.drift_loss_indep import drift_loss
from unet.unet_cfg import InnerModelCFG
from unet.unet import InnerModelNoCFG
from unet.unet_configs import get_inner_model_config

log = logging.getLogger(__name__)

class Denoiser(nn.Module):
    """
    Denoiser based on drifting loss.
    """
    LATENT_KEY = "_latent"

    def __init__(self, cfg):
        super().__init__()
        inner_model_cfg = get_inner_model_config(cfg.model.inner_model_cfg)

        self.cfg_conditioned = False
        unet_spec = getattr(cfg.model, 'unet_1frame_spec', 'base')
        if unet_spec == 'cfg':
            log.info("Using no-text drifting denoiser with accentuated action following")
            self.model = InnerModelCFG(inner_model_cfg)
            self.cfg_conditioned = True
        else:
            log.info("Using no-text drifting denoiser without accentuated action following")
            self.model = InnerModelNoCFG(inner_model_cfg)

        self.temp_list = tuple(cfg.model.temp_list)  # temperatures for the drifting loss
        self.n_neg = cfg.model.n_neg                 # number of negative samples
        self.num_history_steps = inner_model_cfg.num_steps_conditioning
        self.chunk_size = cfg.model.chunk_size       # frames per VAE decode call

        # EMA model
        self.decay = cfg.train.decay
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)        

        ##############################################################################
        ### Drifting loss in latent space
        self.latent_loss_type = getattr(cfg.model, 'latent_loss_type', None)
        assert self.latent_loss_type in (None, "latent_spatial"), f"got unrecognized latent_loss_type {self.latent_loss_type!r}"

        ##############################################################################
        ### Drifting loss in feature space
        self.use_dinov3 = getattr(cfg.model, 'use_dinov3', False)

        if self.use_dinov3:
            from features.dinov3_extractor.extractor import DinoV3FeatureExtractor
            self.dinov3 = DinoV3FeatureExtractor(
                model_name=cfg.dinov3.model_name,
                input_size=cfg.dinov3.input_size,
                block_indices=cfg.dinov3.block_indices,
                repo_dir=cfg.dinov3.get("repo_dir", None),
                weights=cfg.dinov3.get("weights", None),
                source=cfg.dinov3.get("source", "local"),
                motion_mask_alpha=cfg.dinov3.get("motion_mask_alpha", 2.5),
                motion_mask_lambda=cfg.dinov3.get("motion_mask_lambda", None),
                motion_mask_qhi=cfg.dinov3.get("motion_mask_qhi", 0.98),
                motion_mask_tau=cfg.dinov3.get("motion_mask_tau", 0.35),
            )
        else:
            self.dinov3 = None

        self.motion_masking = self.use_dinov3 and self.dinov3.motion_masking_enabled

        ##############################################################################
        ### A VAE is needed to decode latents to pixels whenever DINOv3 is active.
        if self.use_dinov3:
            from diffusers import AutoencoderKL
            log.info("Loading SD3 VAE for grad-enabled feature decoding")
            self.feat_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae")
            self.feat_vae.eval()
            self.feat_vae.requires_grad_(False)
        else:
            self.feat_vae = None

        ##############################################################################
        ### Accentuating action following (action CFG).
        self.cfg_alpha = float(getattr(cfg.model, 'cfg_alpha', 1.0))

        # Training-time alpha sampling distribution
        self.cfg_min = float(getattr(cfg.model, 'cfg_min', 1.0))
        self.cfg_max = float(getattr(cfg.model, 'cfg_max', 2.0))
        self.neg_cfg_pw = float(getattr(cfg.model, 'neg_cfg_pw', 1.0))
        self.no_cfg_frac = float(getattr(cfg.model, 'no_cfg_frac', 0.1))

        if self.cfg_conditioned:
            log.info(f"network-conditioned CFG: cfg_min={self.cfg_min}, cfg_max={self.cfg_max}, "
                     f"neg_cfg_pw={self.neg_cfg_pw}, no_cfg_frac={self.no_cfg_frac}, "
                     f"infer cfg_alpha={self.cfg_alpha}")

    def _sample_cfg(self, b: int, device):
        """
        Sample the guidance scale `alpha` for accentuating action following.
        alpha is drawn from [cfg_min, cfg_max] with density ~ alpha^(-neg_cfg_pw).
        
        Returns:
            cfg: (b,) sampled alphas
            uncond_w: (b,) negative weights = (alpha - 1) * (n_neg - 1)
        """
        frac = torch.rand(b, device=device)
        pw = 1.0 - self.neg_cfg_pw
        if abs(pw) < 1e-6:
            cfg = torch.exp(math.log(self.cfg_min) + frac * (math.log(self.cfg_max) - math.log(self.cfg_min)))
        else:
            cfg = (self.cfg_min ** pw + frac * (self.cfg_max ** pw - self.cfg_min ** pw)) ** (1.0 / pw)
        frac2 = torch.rand(b, device=device)
        cfg = torch.where(frac2 < self.no_cfg_frac, torch.ones_like(cfg), cfg)
        uncond_w = (cfg - 1.0) * (self.n_neg - 1)
        return cfg, uncond_w

    def _run_model(self, model, noise, obs, act, cfg_scale):
        """
        Call the U-Net, passing cfg_scale only when it is CFG-conditioned.
        """
        if self.cfg_conditioned:
            return model(noise, obs, act, cfg_scale)
        return model(noise, obs, act)

    def _cfg_setup(self, b, device):
        """
        Returns the action-CFG state for one forward pass.
        Returns:
            cfg_b: (b,) sampled alphas, or None when the net is not CFG-conditioned
            uncond_w_b: (b,) per-sample negative weights, or None when the net is not CFG-conditioned
            do_uncond: whether to mix the "nothing moved" negative into the loss
        """
        if self.cfg_conditioned:
            cfg_b, uncond_w_b = self._sample_cfg(b, device)
            return cfg_b, uncond_w_b, True
        else:
            return None, None, False

    def _feature_activations(self, latents: torch.Tensor):
        """
        Given a batch of latent frames, returns the token fields that the drifting loss is applied on.
        Args:
            latents: (B, M, C, H, W) latent frames
        Returns:
            dict[str, (B*M, T, D)], one entry per field, where 
                T is the number of tokens per frame and D is the token dimension
              - "blk{i}_tok": DINOv3 patch features (T = number of patches, D = embed dim)
              - LATENT_KEY: one token per latent cell (T = H*W, D = C)
        """
        B, M, C, H, W = latents.shape

        out = {}
        if self.use_dinov3:
            from encoders.vae_utils import decode_latents_with_grad
            # latents (B, M, C, H, W) -> pixels (B, M, 3, H_pix, W_pix) in [-1, 1]
            pix5 = decode_latents_with_grad(
                latents, vae=self.feat_vae, chunk_size=self.chunk_size,
            )
            _, _, Cp, H_pix, W_pix = pix5.shape
            pix = pix5.reshape(B * M, Cp, H_pix, W_pix)
            out.update(self.dinov3.get_activations(pix))

        if self.latent_loss_type == "latent_spatial":
            # one drift field per spatial latent cell: H*W tokens of dimension C
            out[self.LATENT_KEY] = latents.reshape(B * M, C, H * W).transpose(1, 2)

        return out

    def _drift_from_act(self, gen_act: dict, pos_act: dict, B: int, uncond_act: dict = None,
                        uncond_w=None, token_weights: dict = None):
        """
        Apply the drifting loss to a dict of activations.
        The drifting loss is applied independently to each token.

        Args:
            gen_act: dict of (B*N, T, D) generated activations
            pos_act: dict of (B, T, D) positive (data) activations
            B: batch size
            uncond_act: optional dict of (B*U, T, D) unconditional-negative activations (action CFG). U = 1 here.
            uncond_w: (B,) tensor of per-sample negative weights. Required whenever uncond_act is given; both are None when action CFG is off.
            token_weights: optional dict of (B, T) per-token loss multipliers (token-space motion masking).
                    Applied per key; keys absent from it are left unweighted.
        Returns:
            (scalar loss, info dict)
        """
        use_cfg = uncond_act is not None and uncond_w is not None
        loss = 0.0
        info = {}
        for key_name in gen_act.keys():
            _, T, D = pos_act[key_name].shape
            # stopgrad the data samples; only gen_act carries gradient
            p_act = pos_act[key_name].detach().reshape(B*T, 1, D) # (B*T, 1, D)
            g_act = rearrange(gen_act[key_name], '(b n) t d -> (b t) n d', b=B) # (B*T, N, D)

            if use_cfg:
                u_act = rearrange(uncond_act[key_name].detach(), '(b u) t d -> (b t) u d', b=B) # (B*T, U, D)
                U = u_act.shape[1]
                # each sample's negative weight is repeated over its T tokens
                w_neg = uncond_w.to(torch.float32).repeat_interleave(T)[:, None].expand(-1, U).contiguous() # (B*T, U)
            else:
                u_act, w_neg = None, None

            layer_loss, layer_info = drift_loss(gen=g_act, fixed_pos=p_act,
                                                fixed_neg=u_act, weight_neg=w_neg,
                                                R_list=self.temp_list)

            if token_weights is not None and key_name in token_weights:
                w_flat = token_weights[key_name].reshape(-1).to(layer_loss.dtype)
                loss = loss + (layer_loss * w_flat).mean()
            else:
                loss = loss + layer_loss.mean()
            for k, v in layer_info.items():
                info[k] = info.get(k, 0.0) + v
            # No need to divide by the number of keys: gradient norm clipping makes a constant
            # scalar factor on the loss irrelevant.
        return loss, info

    def drifting_loss(self, gen: torch.Tensor, pos: torch.Tensor, uncond: torch.Tensor = None,
                      uncond_w=None, past: torch.Tensor = None):
        """
        Drifting loss: MSE(gen, stopgrad(gen + V)), where V is the drifting field that pulls
        gen towards pos and away from the negatives.

        Two modes:
          - no DINOv3 and no latent-token loss: each frame is one flat C*H*W vector;
          - otherwise: the loss is applied per token (DINOv3 patches and/or latent cells)
            via _feature_activations + _drift_from_act.
        Args:
            gen: Generated samples [B, N, C, H, W]
            pos: Data samples [B, 1, C, H, W]
            uncond: optional unconditional-negative samples [B, U, C, H, W] (action CFG; U=1).
            uncond_w: (B,) tensor of per-sample negative weights. Required whenever uncond is
                given; both are None when action CFG is off.
            past: optional ground-truth past frame o_t [B, 1, C, H, W]. When token-space
                motion weighting is enabled, its features are compared against pos to
                build the per-token loss amplifier.
        Returns:
            scalar drifting loss (the mean over all entries in the batch) and dict of metrics
        """
        use_cfg = uncond is not None and uncond_w is not None
        B, N, C, H, W = gen.shape
        if not self.use_dinov3 and self.latent_loss_type is None:
            # flat single-field latent drift: one drifting field per batch element
            gen_flat = gen.reshape(B, N, -1) # [B, N, D], where D=C*H*W
            pos_flat = pos.reshape(B, 1, -1) # [B, 1, D]
            if use_cfg:
                neg_flat = uncond.reshape(B, -1, gen_flat.shape[-1]) # [B, U, D]
                w_neg = uncond_w.to(torch.float32)[:, None].expand(B, neg_flat.shape[1]).contiguous() # [B, U]
            else:
                neg_flat, w_neg = None, None
            loss, info = drift_loss(gen=gen_flat, fixed_pos=pos_flat,
                                    fixed_neg=neg_flat, weight_neg=w_neg, R_list=self.temp_list)
            return loss.mean(), info

        gen_act = self._feature_activations(gen)   # dict of (B*N, T, D)
        pos_act = self._feature_activations(pos)   # dict of (B, T, D)
        if use_cfg:
            with torch.no_grad():
                uncond_act = self._feature_activations(uncond)  # dict of (B*U, T, D)
        else:
            uncond_act = None

        # Token-space motion weighting: encode the past frame o_t (no grad) and
        # build the per-token loss amplifier from ||z(pos) - z(past)||.
        if self.motion_masking and past is not None:
            with torch.no_grad():
                past_act = self._feature_activations(past)
            token_weights = self.dinov3.motion_token_weights(pos_act, past_act)
        else:
            token_weights = None

        return self._drift_from_act(gen_act, pos_act, B, uncond_act=uncond_act,
                                    uncond_w=uncond_w, token_weights=token_weights)

    def forward(self, batch, device):
        """
        Forward pass with teacher forcing.

        Args:
            batch: dictionary representing a batch of data:
                - video_latents: (B, F, C, H, W) tensor of video latents
                - action: (B, F, action_dim) tensor of actions
            device: device
        Returns:
            (scalar loss averaged over the timesteps, dict of averaged metrics)
        """
        obs = batch['video_latents'].to(device)
        act = batch['action'].to(device)

        b, t, c, h, w = obs.size()
        n = self.num_history_steps # num of timesteps used for history context
        seq_length = t - n # num future frames the model needs to predict in this sequence

        # Get info for accentuating action following
        cfg_b, uncond_w_b, do_uncond = self._cfg_setup(b, device)

        loss = 0
        metrics = defaultdict(float)

        # autoregressive rollout
        #   i=0: previous timesteps 0:n => predict timestep n
        #   i=1: previous timesteps 1:n+1 => predict timestep n+1
        #   i=t-n-1: previous timesteps t-n-1:t-1 => predict timestep t-1
        for i in range(seq_length):
            log.info(f"(iter {i}/{seq_length})")
            prev_obs = obs[:, i : n + i].reshape(b, n * c, h, w)
            prev_act = act[:, i : n + i]
            target_obs = obs[:, n + i].unsqueeze(1) # ground-truth target frame to predict

            # Action CFG: the preceding real frame is the "nothing moved" negative.
            uncond_obs = obs[:, n + i - 1].unsqueeze(1) if do_uncond else None # (b, 1, c, h, w)
            # Past frame o_t for token-space motion masking (ground truth, indep. of CFG).
            past_obs = obs[:, n + i - 1].unsqueeze(1) if self.motion_masking else None

            # Draw k = n_neg samples for the same history by repeating the conditioning.
            prev_obs = prev_obs.repeat_interleave(self.n_neg, dim=0)  # (k*b, n*c, h, w)
            prev_act = prev_act.repeat_interleave(self.n_neg, dim=0)  # (k*b, n, action_dim)

            cfg_scale_i = cfg_b.repeat_interleave(self.n_neg) if self.cfg_conditioned else None

            noise = torch.randn((self.n_neg * b, c, h, w), device=device)

            gen = self._run_model(self.model, noise, prev_obs, prev_act, cfg_scale_i)
            gen = gen.reshape((b, self.n_neg, c, h, w))

            loss_i, info_i = self.drifting_loss(gen, target_obs, uncond=uncond_obs,
                                                uncond_w=uncond_w_b, past=past_obs)
            loss += loss_i

            for key, value in info_i.items():
                metrics[key] += value

        loss /= seq_length
        averages = {key: total / seq_length for key, total in metrics.items()}
        averages['loss_backprop'] = loss.item()
        return loss, averages

    @torch.no_grad()
    def sample(self, prev_obs, prev_act, use_ema=True, init_val=None, cfg_scale=None):
        """
        Sample from DriftWorld. Uses a single forward pass.
        Args:
            prev_obs: (B, N, C, H, W) history observations
            prev_act: (B, N, action_dim) history actions
            use_ema: whether to use the EMA model
            init_val: if given, start from a constant tensor of this value instead of noise
            cfg_scale: guidance scale alpha
        Returns:
            (B, C, H, W) prediction of the next frame
        """
        B, N, C, H, W = prev_obs.size()
        if init_val is None:
            init_tensor = torch.randn((B, C, H, W), device=prev_obs.device)
        else:
            init_tensor = torch.full((B, C, H, W), init_val, device=prev_obs.device)

        obs_flat = prev_obs.reshape((B, N*C, H, W))
        if self.cfg_conditioned:
            alpha = self.cfg_alpha if cfg_scale is None else cfg_scale
            cfg_tensor = torch.full((B,), float(alpha), device=prev_obs.device)
        else:
            cfg_tensor = None

        model = self.model if not use_ema else self.ema_model
        return self._run_model(model, init_tensor, obs_flat, prev_act, cfg_tensor)

    @torch.no_grad()
    def sample_autoregressive_act_len(self, obs, act, use_ema=True, init_val=None, cfg_scale=None):
        """
        Autoregressive rollout
        Args:
            obs: (B, T_obs, C, H, W) frames. Only [0:n] is used, as the history.
            act: (B, T, action_dim) all actions
            use_ema: whether to use EMA
            cfg_scale: guidance scale alpha
        Returns:
            (B, T, 3, H_out, W_out) ground-truth history + predicted future frames in pixel space
        """
        b, _, c, h, w = obs.size()
        t = act.shape[1]
        n = self.num_history_steps
        seq_length = t - n

        # `result` will store the generated video
        result = torch.zeros((b, t, c, h, w), device=obs.device)
        result[:, :n] = obs[:, :n]

        # autoregressive rollout
        #   i=0: previous timesteps 0:n => predict timestep n
        #   i=1: previous timesteps 1:n+1 => predict timestep n+1
        #   i=t-n-1: previous timesteps t-n-1:t-1 => predict timestep t-1
        for i in range(seq_length):
            log.info(f"(iter {i}/{seq_length})")
            prev_obs = result[:, i : n + i]
            prev_act = act[:, i : n + i]
            gen = self.sample(prev_obs, prev_act, use_ema, init_val, cfg_scale=cfg_scale)
            result[:, n + i] = gen

        from encoders.vae_utils import decode_latents
        return decode_latents(result, chunk_size=self.chunk_size)

    @torch.no_grad()
    def sample_not_autoregressive(self, obs, act, use_ema=True, init_val=None, cfg_scale=None):
        """
        Teacher-forced rollout
        Args:
            obs: (B, T, C, H, W) ground-truth frames
            act: (B, T, action_dim) actions
            use_ema: whether to use EMA
            cfg_scale: guidance scale alpha
        Returns:
            (B, T, 3, H_out, W_out) ground-truth history + predicted future frames in pixel space
        """
        assert obs.shape[1] == act.shape[1], f"sample_not_autoregressive is called with obs {obs.shape} and act {act.shape} containing different numbers of frames"
        b, _, c, h, w = obs.size()
        t = act.shape[1]
        n = self.num_history_steps
        seq_length = t - n

        # `result` will store the generated video
        result = torch.zeros((b, t, c, h, w), device=obs.device)
        result[:, :n] = obs[:, :n]

        # teacher-forced rollout:
        #   i=0: previous timesteps 0:n => predict timestep n
        #   i=1: previous timesteps 1:n+1 => predict timestep n+1
        #   i=t-n-1: previous timesteps t-n-1:t-1 => predict timestep t-1
        for i in range(seq_length):
            log.info(f"(iter {i}/{seq_length})")
            prev_obs = obs[:, i : n + i]
            prev_act = act[:, i : n + i]

            gen = self.sample(prev_obs, prev_act, use_ema, init_val, cfg_scale=cfg_scale)
            result[:, n + i] = gen

        from encoders.vae_utils import decode_latents
        return decode_latents(result, chunk_size=self.chunk_size)

    @torch.no_grad()
    def sample_autoregressive(self, obs, act, use_ema=True, init_val=None,
                                       cfg_scale=None, gen_chunk=8):
        """
        Autoregressive rollout of length gen_chunk frames

        Args:
            obs: (B, L, C, H, W) full-episode latents (GT). Not mutated.
            act: (B, L, action_dim) full-episode actions
            use_ema: whether to use EMA
            init_val: if given, start each U-Net call from a constant instead of noise
            cfg_scale: guidance scale alpha
            gen_chunk: number of frames generated per chunk
        Returns:
            (B, L, 3, H_out, W_out) GT history (frames [0:n]) + generated future frames in pixel space
        """
        assert obs.shape[1] == act.shape[1], f"obs {obs.shape} and act {act.shape} contain different numbers of frames"
        b, L, c, h, w = obs.size()
        n = self.num_history_steps
        n_gen_total = L - n
        assert n_gen_total > 0, f"L={L} <= n_history={n}"

        n_chunks = math.ceil(n_gen_total / gen_chunk)
        padded_len = n + n_chunks * gen_chunk

        # Zero-pad obs/act along time so the final chunk is full length (null actions for the tail)
        obs_pad = torch.zeros((b, padded_len, c, h, w), device=obs.device, dtype=obs.dtype)
        obs_pad[:, :L] = obs
        act_pad = torch.zeros((b, padded_len, act.shape[-1]), device=act.device, dtype=act.dtype)
        act_pad[:, :L] = act

        # `result` holds GT history [0:n] then generated frames
        result = torch.zeros((b, padded_len, c, h, w), device=obs.device, dtype=obs.dtype)
        result[:, :n] = obs[:, :n]

        for ci in range(n_chunks):
            s = gen_chunk * ci
            log.info(f"(chunk {ci}/{n_chunks} @ frame {s})")
            win = torch.zeros((b, n + gen_chunk, c, h, w), device=obs.device, dtype=obs.dtype)
            win[:, :n] = obs_pad[:, s : s + n]
            for i in range(gen_chunk):
                prev_obs = win[:, i : n + i]
                prev_act = act_pad[:, s + i : s + n + i]
                gen = self.sample(prev_obs, prev_act, use_ema, init_val, cfg_scale=cfg_scale)
                win[:, n + i] = gen
            result[:, s + n : s + n + gen_chunk] = win[:, n:]

        from encoders.vae_utils import decode_latents
        return decode_latents(result[:, :L], chunk_size=self.chunk_size)

    @torch.no_grad()
    def update_ema(self):
        """
        Updates the EMA parameters.
        """
        for p_ema, p_net in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.lerp_(p_net, 1 - self.decay)
