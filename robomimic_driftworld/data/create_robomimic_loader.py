"""
Dataloader for DriftWorld, built on robomimic's SequenceDataset.

Each sample is a window of consecutive camera frames
    o_{t-num_history}, ..., o_t, ..., o_{t+num_future}
plus the corresponding robot actions.

- The first num_history + 1 frames (o_{t-num_history} ... o_t) are the model input.
- The remaining num_future frames (o_{t+1} ... o_{t+num_future}) are the prediction target.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "robomimic",
    ),
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.dataset import SequenceDataset

AGENTVIEW_CAMERA = "agentview_image"
WRIST_CAMERA = "robot0_eye_in_hand_image"
CAMERAS = (AGENTVIEW_CAMERA, WRIST_CAMERA)
MAIN_CAMERA = AGENTVIEW_CAMERA


def _init_obs_utils(cameras=(AGENTVIEW_CAMERA,)):
    """
    Set observation modalities so robomimic knows the given camera keys are RGB images.
    This needs to be run before constructing a SequenceDataset.
    """
    ObsUtils.initialize_obs_utils_with_obs_specs(
        {
            "obs": {
                "rgb": list(cameras),
                "low_dim": [],
            }
        }
    )


def _resize_frames(frames, new_size):
    """
    Resize a stack of uint8 frames to (new_size, new_size).
    Args:
        frames (np.ndarray): (T, H, W, 3) uint8 image sequence
        new_size (int): target height and width
    Returns:
        np.ndarray: (T, new_size, new_size, 3) uint8 image sequence
    """
    t = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(new_size, new_size), mode="bilinear", align_corners=False)
    t = t.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    return t.numpy()


class _ProcessObsWrapper(Dataset):
    """
    Wraps a SequenceDataset and processes (resize + normalize) one or more image observations
    so the DataLoader collates already-processed frames.
    """

    def __init__(self, dataset, obs_keys, new_size, normalize_img):
        self.dataset = dataset
        self.obs_keys = tuple(obs_keys)
        self.new_size = new_size
        self.normalize_img = normalize_img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        for obs_key in self.obs_keys:
            frames = item["obs"][obs_key]
            if self.new_size is not None:
                frames = _resize_frames(frames, self.new_size)
            if self.normalize_img:
                frames = frames.astype(np.float32) / 255.0
                frames = (frames - 0.5) / 0.5
            frames = np.transpose(frames, (0, 3, 1, 2)) # (T, H, W, C) -> (T, C, H, W)
            item["obs"][obs_key] = frames
        return item


def get_dataset(
    hdf5_path,
    num_history=3,
    num_future=2,
    new_size=None,
    hdf5_cache_mode="low_dim",
    filter_by_attribute=None,
    normalize_img=True,
    cameras=(AGENTVIEW_CAMERA,),
):
    """
    Build a SequenceDataset that yields windows of camera frames + actions.

    Args:
        hdf5_path (str): path to the robomimic image hdf5
        num_history (int): number of past frames before the current frame (input)
        num_future (int): number of future frames to predict (target)
        new_size (int): If provided, resize each frame to (new_size, new_size). If None, keep the raw 84x84.
        hdf5_cache_mode (str): "low_dim" (only cache low dim; images are read from files)
                               "all" (cache everything)
                               None (no caching)
        filter_by_attribute (str): "train", "val", or None
        normalize_img: whether to normalize image to [-1, 1] range
        cameras (tuple): camera keys to load (default: the single agentview camera)

    Returns:
        Dataset whose samples contain:
            obs[cam]: (num_history + num_future + 1, 3, S, S) float32 tensor for each cam in cameras,
                      where S = new_size if given else 84
            actions: (num_history + num_future + 1, 7) float32 tensor
    """
    cameras = tuple(cameras)
    _init_obs_utils(cameras)
    dataset = SequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=cameras,                             # one or more camera views
        action_keys=("actions",),                     # 7-dim action
        dataset_keys=("actions",),
        action_config={"actions": {"normalization": None}},
        frame_stack=num_history + 1,                  # num_history previous frames + 1 current frame
        seq_length=num_future + 1,                    # num_future future frames + 1 current frame
        pad_frame_stack=True,                         # enable padding history frames by repeating
        pad_seq_length=False,                         # disable padding for future frames b/c they should always be real
        get_pad_mask=True,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        load_next_obs=False,
        filter_by_attribute=filter_by_attribute,
    )
    if new_size is not None or normalize_img:
        dataset = _ProcessObsWrapper(dataset, obs_keys=cameras, new_size=new_size, normalize_img=normalize_img)
    return dataset


def get_dataloader(cfg, rank=0, world_size=1):
    """
    Returns robomimic dataloader. The camera views are selected by cfg.model.unet_type
    ("multi_2view" loads agentview + wrist; otherwise agentview only).
    """
    cameras = (AGENTVIEW_CAMERA, WRIST_CAMERA) if cfg.model.unet_type == "multi_2view" else (AGENTVIEW_CAMERA,)
    dataset = get_dataset(
        cfg.data.hdf5_path,
        num_history=cfg.data.num_history,
        num_future=cfg.data.num_future,
        new_size=cfg.data.new_size,
        hdf5_cache_mode=cfg.data.hdf5_cache_mode,
        filter_by_attribute=cfg.data.filter_by_attribute,
        normalize_img=cfg.data.normalize_img,
        cameras=cameras,
    )

    sampler = None
    shuffle = cfg.dataloader.shuffle
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=cfg.dataloader.shuffle,
            drop_last=True,
        )
        shuffle = False  # sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
        sampler=sampler,
        shuffle=shuffle,
        pin_memory=cfg.dataloader.pin_memory,
        persistent_workers=True,
        drop_last=True,
    )
