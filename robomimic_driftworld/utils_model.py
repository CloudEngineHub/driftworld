from drifting_denoiser_multi import Denoiser

def create_model(cfg, device):
    return Denoiser(unet_type=cfg.model.unet_type, temp_list=cfg.model.temp_list,
                        n_neg=cfg.model.n_neg, num_future_frames=cfg.model.num_future_frames,
                        num_history_frames=cfg.model.num_history_frames,
                        decay=cfg.train.decay).to(device)
