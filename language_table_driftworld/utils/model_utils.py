def create_model(cfg, device):
    from drifting_denoiser import Denoiser
    return Denoiser(cfg).to(device)

def create_model_selfforce(cfg, device):
    from drifting_denoiser_selfforce import Denoiser
    return Denoiser(cfg).to(device)
