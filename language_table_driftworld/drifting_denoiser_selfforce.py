"""
DriftWorld denoiser for Language Table: phase 2 self-forcing
"""
from drifting_denoiser import Denoiser as BaseDenoiser

class Denoiser(BaseDenoiser):

    def phase2_prepare(self, batch, device):
        obs = batch['video_latents'].to(device)
        act = batch['action'].to(device)

        b, t, c, h, w = obs.size()
        n = self.num_history_steps
        seq_length = t - n
        k = self.n_neg

        rollout = obs.repeat_interleave(k, dim=0)        # (k*b, t, c, h, w)
        act_k = act.repeat_interleave(k, dim=0)          # (k*b, t, action_dim)

        return dict(
            obs=obs, rollout=rollout, act_k=act_k,
            b=b, n=n, c=c, h=h, w=w, k=k, seq_length=seq_length,
        )

    def phase2_step(self, prev_obs, prev_act, noise, target_obs, past_obs, b, k):
        """
        One self-forcing step.

        Returns:
            loss_i: scalar drift loss for this step
            gen_det: (k*b, c, h, w) detached prediction to write into rollout[:, n+i]
            info_i: metrics dict
        """
        gen = self.model(noise, prev_obs, prev_act)  # (k*b, c, h, w)
        c, h, w = gen.shape[-3], gen.shape[-2], gen.shape[-1]
        gen5 = gen.reshape(b, k, c, h, w)
        loss_i, info_i = self.drifting_loss(gen5, target_obs, past=past_obs)
        return loss_i, gen.detach(), info_i
