"""
DriftWorld: Drifting denoiser, which contains the forward pass, drifting loss, and sampling code.
Supports one or more camera views.
"""

import torch
import torch.nn as nn
import logging
from collections import defaultdict
import copy
import math

from unet.unet_configs import UNet_model_dict
from drifting.drift_loss_indep import drift_loss

log = logging.getLogger(__name__)

# Camera view keys
AGENTVIEW_CAMERA = "agentview_image"
WRIST_CAMERA = "robot0_eye_in_hand_image"

class Denoiser(nn.Module):
    """
    Denoiser based on drifting loss.
    It conditions on num_history_frames current+history frames and generates num_future_frames future frames.
    """
    def __init__(self, unet_type: str, # type of U-Net
                 temp_list: tuple, # temperatures for drifting loss
                 n_neg: int, # number of negative samples for drifting loss
                 num_future_frames: int, # number of future frames to predict
                 num_history_frames: int, # number of current+history frames to condition on
                 decay: float, # EMA decay
                 ) -> None:
        super().__init__()

        if unet_type == "multi":
            self.cameras = (AGENTVIEW_CAMERA,)
        elif unet_type == "multi_2view":
            self.cameras = (AGENTVIEW_CAMERA, WRIST_CAMERA)
        else:
            raise NotImplementedError("Invalid model type")

        self.inner_model = UNet_model_dict[unet_type](num_history=num_history_frames)
        self.temp_list = temp_list
        self.n_neg = n_neg
        self.num_future_frames = num_future_frames
        self.num_history_frames = num_history_frames
        self.decay = decay
        self.img_channels = 3 * len(self.cameras)
        self.ema_model = copy.deepcopy(self.inner_model)
        self.ema_model.requires_grad_(False)

    def _stack_views(self, batch, device):
        """
        Read the configured camera views and concatenate them channelwise.
        Args:
            batch: dict with batch["obs"][cam] for each cam in self.cameras, each (B, T, 3, H, W).
        Returns:
            (B, T, 3*len(self.cameras), H, W) tensor
        """
        views = [batch["obs"][cam].to(device) for cam in self.cameras]
        return torch.cat(views, dim=2)

    def drifting_loss(self, gen: torch.Tensor, pos: torch.Tensor):
        """
        Drifting loss: MSE(gen, stopgrad(gen + V)).
        Args:
            gen: Generated samples [B, N, C, H, W]
            pos: Data samples [B, 1, C, H, W]
        Returns:
            scalar drifting loss and dictionary of metrics
        """
        B, N, C, H, W = gen.shape
        gen_flat = gen.reshape(B, N, -1) # [B, N, D], where D=C*H*W
        pos_flat = pos.reshape(B, 1, -1) # [B, 1, D]

        loss, info = drift_loss(gen=gen_flat, fixed_pos=pos_flat, R_list=self.temp_list)
        return loss.mean(), info

    def forward(self, batch, device):
        """
        Forward pass to train DriftWorld
        Args:
            batch: a batch of data containing
            - batch["obs"][cam] for each cam in self.cameras: visual observations
                                               o_(t-h), ..., o_t, o_(t+1), ..., o_(t+n)
                                               each of shape (B, h+n+1, 3, 96, 96)
            - batch["actions"]: actions a_(t-h), ..., a_(t+n)
                                shape (B, h+n+1, 7)
            device: device
        Returns:
            scalar loss and dictionary of metrics
        """
        obs = self._stack_views(batch, device)  # (B, T, 3*num_cameras, H, W)
        act = batch["actions"].to(device)

        b, t, c, h, w = obs.size()
        assert c == self.img_channels, f"expected {self.img_channels} channels, got {c}"
        n = self.num_future_frames # num of future frames output by one forward pass
        k = self.num_history_frames # num of current+history frames used as context
        cur_idx = k - 1 # index of the current frame o_t within the window
        assert t == n + k

        target_x = obs[:, cur_idx+1:].permute(0, 2, 1, 3, 4) # (b, n, c, h, w) -> (b, c, n, h, w)
        history = obs[:, 0 : cur_idx+1].permute(0, 2, 1, 3, 4) # (b, k, c, h, w) -> (b, c, k, h, w)
        actions = act[:, cur_idx : cur_idx+n] # (b, n, 7)

        history = history.repeat_interleave(self.n_neg, dim=0)
        actions = actions.repeat_interleave(self.n_neg, dim=0)

        # j = self.n_neg
        # gen: (j*b, c, n, h, w) generated samples, i.e. "negative" samples for drifting
        noise = torch.randn((self.n_neg * b, c, n, h, w), device=device)
        gen = self.inner_model(noise, history, actions)
            # gen is the output from the U-Net. The inputs are
            # noise: (j*b, c, n, h, w) noise for future states s_(T+1), ..., s_(T+n)
            # history: (j*b, c, k, h, w) current+history states s_(T-k+1), ..., s_T
            # actions: (j*b, n, 7) actions a_T, ..., a_(T+n-1)

        # compute drifting loss for every timestep separately
        loss = 0
        metrics = defaultdict(float)
        for i in range(n):
            target_x_slice = target_x[:, :, i].reshape((b, 1, c, h, w)) # (b, c, h, w) -> (b, 1, c, h, w)
            gen_slice = gen[:, :, i].reshape((b, self.n_neg, c, h, w)) # (j*b, c, h, w) -> (b, j, c, h, w)
            loss_i, info_i = self.drifting_loss(gen_slice, target_x_slice)
            loss += loss_i
            for key, value in info_i.items():
                metrics[key] += value

        loss /= n
        averages = {key: total / n for key, total in metrics.items()}
        averages['loss_backprop'] = loss.item()
        return loss, averages

    @torch.no_grad()
    def sample(self, history, actions):
        """
        Sample from DriftWorld.
        Args:
            history: (B, C, K, H, W) tensor of the current+history states o_(t-K+1), ..., o_t
            actions: (B, n, 7) tensor of actions a_t, ..., a_(t+n-1)
        Returns:
            (B, n, C, H, W) tensor of predicted future states o_(t+1), ..., o_(t+n)
        """
        B, C, K, H, W = history.size()
        n = actions.shape[1]
        init_tensor = torch.randn((B, C, n, H, W), device=history.device)
        return self.ema_model(init_tensor, history, actions).permute(0, 2, 1, 3, 4)

    @torch.no_grad()
    def sample_autoregressive(self, cur_state, actions):
        """
        Sample autoregressively from DriftWorld.
        Args:
            cur_state: (B, h+F+1, C, H, W) tensor of o_(t-h), ..., o_t, o_(t+1), ..., o_(t+F)
                        We only need to know the history frames o_(t-h), ..., o_t. The model will predict the future frames.
            actions: (B, h+F+1, 7) actions a_(t-h), ..., a_(t+F)
        Returns:
            (B, h+F+1, C, H, W) tensor of h+1 ground-truth and F predicted frames
        """
        h = self.num_history_frames - 1 # also equals index of the current frame o_t within the window
        n = self.num_future_frames # number of future frames the model generates in a single forward pass
        F = actions.shape[1] - h - 1 # number of future frames to predict

        num_iter = math.ceil(F / n)
        log.info(f"Number future frames: F = {F}")
        log.info(f"Number of iterations: num_iter = {num_iter}")

        # autoregressive rollout
        out = torch.zeros_like(cur_state)
        out[:, :h+1] = cur_state[:, :h+1] # fill in ground-truth history o_(t-h), ..., o_t
        for i in range(num_iter):
            log.info(f"(iter {i}/{num_iter})")
            history_i = out[:, i*n : i*n + h + 1].permute(0, 2, 1, 3, 4) # (B, h+1, C, H, W) -> (B, C, h+1, H, W)
            act_i = actions[:, h + i*n : h + (i+1)*n] # (B, n, 7)
            gen = self.sample(history_i, act_i) # (B, n, C, H, W)
            out[:, h + i*n + 1 : h + i*n + 1 + gen.shape[1]] = gen
        return out

    @torch.no_grad()
    def update_ema(self):
        """
        Updates the EMA parameters
        """
        for p_ema, p_net in zip(self.ema_model.parameters(), self.inner_model.parameters()):
            p_ema.lerp_(p_net, 1 - self.decay)
