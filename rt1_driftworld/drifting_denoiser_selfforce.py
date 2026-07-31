"""
DriftWorld denoiser for RT-1: phase 2 self-forcing
"""
from drifting_denoiser import Denoiser as BaseDenoiser


class Denoiser(BaseDenoiser):

    def phase2_prepare(self, batch, device):
        """
        Setup for phase 2
        Returns:
            dict with everything the per-step loop needs (buffers, CFG state, and shapes)
        """
        obs = batch['video_latents'].to(device)
        act = batch['action'].to(device)

        b, t, c, h, w = obs.size()
        n = self.num_history_steps
        seq_length = t - n
        k = self.n_neg

        rollout = obs.repeat_interleave(k, dim=0)        # (k*b, t, c, h, w)
        act_k = act.repeat_interleave(k, dim=0)          # (k*b, t, action_dim)

        cfg_b, uncond_w_b, do_uncond = self._cfg_setup(b, device)
        cfg_scale_k = cfg_b.repeat_interleave(k) if self.cfg_conditioned else None

        return dict(
            obs=obs, rollout=rollout, act_k=act_k,
            cfg_scale_k=cfg_scale_k, uncond_w_b=uncond_w_b, do_uncond=do_uncond,
            b=b, n=n, c=c, h=h, w=w, k=k, seq_length=seq_length,
        )

    def phase2_step(self, prev_obs, prev_act, cfg_scale_k, noise,
                    target_obs, uncond_obs, uncond_w_b, past_obs, b, k):
        """
        One self-forcing step.

        Args:
            prev_obs: (k*b, n*c, h, w) history latents (detached)
            prev_act: (k*b, n, action_dim) history actions
            cfg_scale_k: (k*b,) guidance scales, or None when the net is not CFG-conditioned
            noise: (k*b, c, h, w) input noise
            target_obs: (b, 1, c, h, w) ground-truth next frame
            uncond_obs: (b, 1, c, h, w) "nothing moved" negative, or None
            uncond_w_b: (b,) per-sample negative weights, or None
            past_obs: (b, 1, c, h, w) past frame for motion masking, or None
            b, k: batch size and number of samples per history (k = n_neg)
        Returns:
            loss_i:  scalar drift loss for this step
            gen_det: (k*b, c, h, w) detached prediction to write into rollout[:, n+i]
            info_i:  metrics dict
        """
        gen = self._run_model(self.model, noise, prev_obs, prev_act, cfg_scale_k)  # (k*b, c, h, w)
        c, h, w = gen.shape[-3:]
        gen5 = gen.reshape(b, k, c, h, w)
        loss_i, info_i = self.drifting_loss(gen5, target_obs, uncond=uncond_obs,
                                            uncond_w=uncond_w_b, past=past_obs)
        return loss_i, gen.detach(), info_i
