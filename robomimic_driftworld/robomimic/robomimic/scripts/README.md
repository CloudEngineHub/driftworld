# Original Dataset
After running `python scripts/get_dataset_info.py --dataset datasets/lift/mh/image.hdf5`: The dataset layout is
```
data/                          (attrs: env_args = env config JSON)
  demo_0/                       (attrs: num_samples = T, the episode length)
    obs/
      agentview_image           (T, 84, 84, 3)  uint8
      robot0_eye_in_hand_image  (T, 84, 84, 3)  uint8
      robot0_eef_pos            (T, 3)   float
      robot0_eef_quat           (T, 4)
      robot0_gripper_qpos       (T, 2)
      object                    (T, D)   object poses
    next_obs/                   # same keys, shifted by one timestep (s')
    actions                     (T, 7)   # OSC_POSE: 3 dpos + 3 drot + 1 gripper, in [-1, 1]
    rewards                     (T,)
    dones                       (T,)
    states                      (T, S)   # raw MuJoCo sim state
  demo_1/
  ...
mask/                           # filter keys: train / valid split, etc.
```
Key facts for the standard robomimic image datasets:

- Images are 84×84×3, uint8 (0–255). Two camera views: agentview (fixed) and robot0_eye_in_hand (wrist).
- Each demo is variable length T (lift mh averages ~50 steps).
- Actions are 7-dim, normalized to [−1, 1].
- obs at time t and next_obs at time t together give you the (s_t, a_t, s_{t+1}) transition.

# What is needed for action-conditioned world model
To build one training sample with context length `k` and horizon `h`, you slice within a single demo:

- Input frames: `obs/agentview_image[t-k+1 : t+1]` → context of `k` frames
- Input actions: `actions[t : t+h]` → the `h` actions taken from `t` onward
- Target frames: `obs/agentview_image[t+1 : t+h+1]` → the `h` future frames to predict