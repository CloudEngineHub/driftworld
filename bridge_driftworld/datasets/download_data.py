"""
Based on https://github.com/world-model-eval/world-model-eval/blob/master/src/world_model_eval/download_data.py
which was derived from https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb
"""

import argparse
import functools
import hashlib
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import imageio.v3 as iio
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from decord import VideoReader, cpu
from tqdm import tqdm

BRIDGE_V2_PATH = "./bridge/raw/rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0"

def map_observation(
    to_step: Dict[str, Any],
    from_step: Dict[str, Any],
    from_image_feature_names: Tuple[str, ...] = ("image",),
    to_image_feature_names: Tuple[str, ...] = ("image",),
) -> None:
    """
    Maps specific observation features from the raw dataset step to the target step dictionary.
    
    Args:
        to_step: The destination dictionary where mapped observations will be stored.
        from_step: The source dictionary containing raw step data from TFDS.
        from_image_feature_names: Keys to extract from the source observation dict.
        to_image_feature_names: Corresponding keys to populate in the target observation dict.
    """
    for from_feature_name, to_feature_name in zip(
        from_image_feature_names, to_image_feature_names
    ):
        to_step["observation"][to_feature_name] = from_step["observation"][
            from_feature_name
        ]


def terminate_bool_to_act(terminate_episode: np.ndarray) -> np.ndarray:
    """
    Converts a boolean episode termination flag into a one-hot action vector.
    
    Args:
        terminate_episode: A boolean or binary flag indicating episode termination.
        
    Returns:
        np.ndarray: A one-hot encoded termination vector of shape (3,).
    """
    if terminate_episode == 1.0:
        return np.array([1, 0, 0], dtype=np.int32)
    else:
        return np.array([0, 1, 0], dtype=np.int32)


def rescale_action_with_bound(
    actions: np.ndarray,
    low: float,
    high: float,
    safety_margin: float = 0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> np.ndarray:
    """
    Rescales an action array from its original bounds to a new target range.
    
    Args:
        actions: The raw action array. Expected shape: (D,).
        low: The original lower bound of the action space.
        high: The original upper bound of the action space.
        safety_margin: Margin to avoid hitting the absolute minimum/maximum.
        post_scaling_max: The target upper bound.
        post_scaling_min: The target lower bound.
        
    Returns:
        np.ndarray: The rescaled and clipped action array. Shape: (D,).
    """
    resc_actions = (actions - low) / (high - low) * (
        post_scaling_max - post_scaling_min
    ) + post_scaling_min
    return tf.clip_by_value(
        resc_actions,
        post_scaling_min + safety_margin,
        post_scaling_max - safety_margin,
    )


def _rescale_action(
    action: Dict[str, np.ndarray],
    wv_lo: float = -0.05,
    wv_hi: float = 0.05,
    rd_lo: float = -0.25,
    rd_hi: float = 0.25,
) -> Dict[str, np.ndarray]:
    """
    Rescales the translation and rotation components of a Bridge action dictionary.
    
    Args:
        action: A dictionary containing 'world_vector' of shape (3,) and 
            'rotation_delta' of shape (3,).
        wv_lo, wv_hi: Original bounds for the world vector (translation).
        rd_lo, rd_hi: Original bounds for the rotation delta.
        
    Returns:
        Dict[str, np.ndarray]: The updated action dictionary with rescaled arrays.
            'world_vector' shape: (3,), 'rotation_delta' shape: (3,).
    """
    # Rescale world vector
    action["world_vector"] = rescale_action_with_bound(
        action["world_vector"],
        low=wv_lo,
        high=wv_hi,
        safety_margin=0.01,
        post_scaling_max=1.75,
        post_scaling_min=-1.75,
    )

    # Rescale rotation delta
    action["rotation_delta"] = rescale_action_with_bound(
        action["rotation_delta"],
        low=rd_lo,
        high=rd_hi,
        safety_margin=0.01,
        post_scaling_max=1.4,
        post_scaling_min=-1.4,
    )

    return action


def bridge_v2_map_action(
    to_step: Dict[str, Any],
    from_step: Dict[str, Any],
) -> None:
    """
    Maps and formats a raw Bridge V2 action into the standard step format.
    
    Extracts translation (3,), rotation (3,), and gripper state (1,), maps the gripper 
    to [-1.0, 1.0], and scales the continuous actions to standardized bounds.
    
    Args:
        to_step: The destination dictionary to populate.
        from_step: The source dictionary containing raw Bridge V2 step data.
    """
    to_step["action"]["terminate_episode"] = terminate_bool_to_act(
        from_step["is_terminal"]
    )
    to_step["action"]["world_vector"] = from_step["action"][0:3]
    to_step["action"]["rotation_delta"] = from_step["action"][3:6]

    open_gripper = from_step["action"][6:7]
    open_gripper = tf.round(open_gripper)
    open_gripper = -(open_gripper * 2 - 1)
    to_step["action"]["gripper_closedness_action"] = open_gripper

    to_step["action"] = _rescale_action(to_step["action"])


bridge_v2_map_observation = functools.partial(
    map_observation,
    from_image_feature_names=("image_0",),
    to_image_feature_names=("image",),
)


def _extract_instruction(steps_iter) -> str:
    # Bridge instructions are constant per episode; read once from the first step.
    first = next(iter(steps_iter))
    obs = first.get("observation", {}) if isinstance(first, dict) else {}
    candidates = [
        first.get("language_instruction") if isinstance(first, dict) else None,
        first.get("natural_language_instruction") if isinstance(first, dict) else None,
        obs.get("natural_language_instruction") if isinstance(obs, dict) else None,
        obs.get("language_instruction") if isinstance(obs, dict) else None,
    ]
    for c in candidates:
        if c is None:
            continue
        if isinstance(c, bytes):
            return c.decode("utf-8", errors="replace")
        if isinstance(c, np.ndarray):
            return c.tobytes().decode("utf-8", errors="replace").rstrip("\x00")
        return str(c)
    return ""


def episode_map_fn(
    episode: Dict[str, Any],
    map_step: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Processes a complete episode by applying a mapping function to each step.
    Stacks the individual step frames and actions into episode-length tensors.

    Args:
        episode: A dictionary representing an entire episode with a list of "steps".
        map_step: A callable that formats an individual step.
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'video': Stacked image frames of shape (T, H, W, C).
            - 'action': Stacked action vectors of shape (T, D).
            - 'instruction': Natural-language task instruction string for the episode.
    """
    raw_steps = list(episode["steps"])
    instruction = _extract_instruction(raw_steps)
    steps = list(map(map_step, raw_steps))
    frames = np.stack([s["observation"]["image"] for s in steps], axis=0)
    return {
        "video": frames,
        "action": np.stack([s["action"] for s in steps]),
        "instruction": instruction,
    }


def step_map_fn(
    step: Dict[str, Any], map_observation: Callable, map_action: Callable
) -> Dict[str, Any]:
    """
    Standardizes a single environment step into a uniform observation and 10D action space.
    
    Zero-pads unused action dimensions (e.g., base displacement) to ensure all 
    datasets conform to a consistent 10-dimensional action vector.
    
    Args:
        step: Raw TFDS step dictionary.
        map_observation: Callable to map the observation features.
        map_action: Callable to map and extract the specific dataset's actions.
    Returns:
        Dict[str, Any]: The transformed step. 'action' is a concatenated np.ndarray of shape (10,).
    """
    transformed_step = {}

    transformed_step["observation"] = {}
    transformed_step["action"] = {
        "gripper_closedness_action": np.zeros(1, dtype=np.float32),
        "rotation_delta": np.zeros(3, dtype=np.float32),
        "terminate_episode": np.zeros(3, dtype=np.int32),
        "world_vector": np.zeros(3, dtype=np.float32),
    }

    # Apply dataset-specific mappings
    map_observation(transformed_step, step)
    map_action(transformed_step, step)

    # Concatenate action components into single vector
    action = np.concatenate(
        [
            transformed_step["action"]["world_vector"],
            transformed_step["action"]["rotation_delta"],
            transformed_step["action"]["gripper_closedness_action"],
        ],
        axis=0,
    )
    transformed_step["action"] = action

    return transformed_step


def get_dataset_configs(dataset_home: str) -> Dict[str, Dict[str, Any]]:
    """
    Constructs the configuration dictionary for the Bridge V2 dataset.

    Args:
        dataset_home: Path to the Bridge TFDS builder directory (the folder containing
            dataset_info.json, features.json, and the bridge_dataset-*.tfrecord-* shards).
    Returns:
        Dict[str, Dict[str, Any]]: Configuration map containing the TFDS builder
            directory and the specific episode mapping function.
    """
    return {
        "bridge_v2": {
            "builder_dir": dataset_home,
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=bridge_v2_map_observation,
                map_action=functools.partial(bridge_v2_map_action),
            ),
        },
    }


def _write_episode(
    base_path: Path,
    episode: Dict[str, Any],
    fps: int,
    overwrite: bool,
) -> Dict[str, Any]:
    """
    Write one episode's artifacts and return its manifest row.

    Always writes {stem}.mp4 + .npz + .txt.
    Each artifact is gated `if overwrite or not exists`, so a resumed run only fills in
    what is missing. The {stem}.meta.json sidecar (the manifest row) is written LAST,
    atomically (temp + os.replace), so its presence is the "episode fully complete"
    sentinel for resume.

    Enforces that the encoded mp4 decodes to exactly the source frame count, and
    fingerprints the RAW (pre-encode) frames with sha1.
    """
    video = episode["video"] # (T, H, W, C) uint8
    action = episode["action"]

    # Hard guarantee: video / action must describe the same T-step episode.
    if action.shape[0] != video.shape[0]:
        raise RuntimeError(
            f"{base_path.name}: action len {action.shape[0]} != "
            f"video frames {video.shape[0]}"
        )

    # Content fingerprint of the RAW (pre-encode) frames -- deterministic
    video_sha1 = hashlib.sha1(np.ascontiguousarray(video).tobytes()).hexdigest()

    mp4_path = str(base_path) + ".mp4"
    if overwrite or not os.path.exists(mp4_path):
        iio.imwrite(mp4_path, video, fps=fps, codec="libx264")
        # Read-back verification: the encoder must round-trip the exact frame count.
        decoded_n = len(VideoReader(mp4_path, ctx=cpu(0), num_threads=1))
        if decoded_n != video.shape[0]:
            raise RuntimeError(
                f"{base_path.name}: encoded mp4 decodes to {decoded_n} frames "
                f"but source video has {video.shape[0]}"
            )

    npz_path = str(base_path) + ".npz"
    if overwrite or not os.path.exists(npz_path):
        np.savez(npz_path, action)

    txt_path = str(base_path) + ".txt"
    if overwrite or not os.path.exists(txt_path):
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(episode["instruction"])

    row = {
        "index": base_path.name,
        "n_frames": int(video.shape[0]),
        "video_sha1": video_sha1,
        "instruction": episode["instruction"],
    }

    # Sidecar manifest row
    meta_path = str(base_path) + ".meta.json"
    tmp_path = meta_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(row, f)
    os.replace(tmp_path, meta_path)

    return row


def _assemble_manifest(split_output_dir: Path, split: str) -> None:
    """
    Assemble manifest_{split}.jsonl from every {i:09d}.meta.json sidecar in the split dir.

    Globs + sorts by stem and writes one row per episode in index order, so the manifest is
    byte-identical regardless of how many runs / shards produced the sidecars. Safe to call repeatedly.
    """
    meta_files = sorted(split_output_dir.glob("*.meta.json"))
    rows = []
    for mf in meta_files:
        try:
            rows.append(json.loads(mf.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"WARNING: skipping unreadable sidecar {mf}: {e} (will be regenerated next run)")
    if rows:
        manifest_path = split_output_dir / f"manifest_{split}.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(rows)} manifest rows -> {manifest_path}")


def convert_dataset(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    output_dir: str,
    fps: int,
    max_episodes: int | None = None,
    splits: Tuple[str, ...] = ("train", "val"),
    overwrite: bool = False,
    num_shards: int = 1,
    shard_id: int = 0,
    finalize_manifest: bool = False,
) -> None:
    """
    Downloads, processes, and exports a TFDS dataset into MP4 videos and NPZ action arrays.

    Iterates through dataset splits, applies the dataset-specific mapping functions,
    and saves paired .mp4 (video frames) and .npz (action tensors) files.

    Args:
        dataset_name: Name of the dataset to process (e.g., "bridge_v2").
        dataset_config: Configuration dict containing builder paths and mapping functions.
        output_dir: The directory where the processed datasets will be saved.
        fps: The framerate metadata to embed into the generated MP4 files.
        overwrite: If True, force re-encode/rewrite every artifact (incl. sidecars), ignoring
            existing files. If False (default), keep existing files and skip episodes whose
            full artifact set already exists (idempotent resume).
        num_shards: Split each TFDS split into this many contiguous global-index blocks so K
            machines can run in parallel into one shared output_dir. Each episode is written at
            its TRUE global index, so the union across shards is identical to a single-machine
            run. All other flags + the TFDS build must match across shards for IDs to line up.
        shard_id: Which block (0 <= shard_id < num_shards) this process handles.
        finalize_manifest: If True, do NO TFDS work -- just assemble manifest_{split}.jsonl from
            existing sidecars for each split, then return. Run this once (any machine) after all
            shards finish, since per-shard runs (num_shards>1) skip the shared manifest write.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")

    print(f"Converting {dataset_name}...")

    # Create output directories
    dataset_output_dir = Path(output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Finalize-only mode: assemble manifests from existing sidecars, no dataset build/read.
    if finalize_manifest:
        print("Finalize-manifest mode: assembling manifests from existing sidecars...")
        for split in splits:
            _assemble_manifest(dataset_output_dir / split, split)
        return

    # Build dataset
    print("Building dataset...")
    try:
        dataset_builder = tfds.builder_from_directory(
            builder_dir=dataset_config["builder_dir"]
        )
    except Exception as e:
        print(f"Failed to build dataset {dataset_name}: {e}")
        return

    print("Dataset built successfully.")

    # Process each split
    for split in splits:
        split_output_dir = dataset_output_dir / split
        split_output_dir.mkdir(exist_ok=True)

        # Partition [0, total) into num_shards contiguous blocks; this process handles its own
        # [start, end). max_episodes bounds the global total.
        # Each episode is written at its true global index start+j, so the union over shards matches
        # a single-machine run exactly. TFDS absolute-index slicing reads only the ~1/K shards
        # covering [start, end), giving a real parallel speedup (not just skipping).
        try:
            num_examples = int(dataset_builder.info.splits[split].num_examples)
        except Exception as e:
            print(f"Split {split} not available for {dataset_name}: {e}")
            continue
        total = num_examples if max_episodes is None else min(num_examples, max_episodes)
        start = shard_id * total // num_shards
        end = (shard_id + 1) * total // num_shards
        if start >= end:
            print(f"Nothing to do for {split} shard {shard_id}/{num_shards} "
                  f"(range [{start}, {end}) of {total}).")
            continue
        print(f"{split}: total={total} | shard {shard_id}/{num_shards} -> "
              f"global range [{start}, {end}) ({end - start} episodes)")

        try:
            # Deterministic ordering. The {i:09d} index assigned below is the
            # GLOBAL index start+j, which must be identical across runs/devices/shards (given
            # the same TFDS build) so every stem-keyed artifact -- mp4, npz, txt,
            # precomputed latents -- lines up.
            #   - split[start:end]: absolute-index slice
            #   - shuffle_files=False: read shards in fixed order, no per-run shuffle
            #   - interleave_cycle_length/block_length=1: read shards strictly sequentially
            read_config = tfds.ReadConfig(
                interleave_cycle_length=1,
                interleave_block_length=1,
            )
            dataset = dataset_builder.as_dataset(
                split=f"{split}[{start}:{end}]", shuffle_files=False, read_config=read_config
            )
        except Exception as e:
            print(f"Split {split} not available for {dataset_name}: {e}")
            continue

        # Force order-preserving execution before prefetch
        options = tf.data.Options()
        options.experimental_deterministic = True
        dataset = dataset.with_options(options)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = tfds.as_numpy(dataset)

        print(f"Processing {split} split...")
        for j, episode in tqdm(enumerate(dataset), desc=f"{dataset_name}-{split}"):
            gidx = start + j  # True global episode index
            try:
                base_path = split_output_dir / f"{gidx:09d}"
                required = [
                    str(base_path) + ".mp4",
                    str(base_path) + ".npz",
                    str(base_path) + ".txt",
                    str(base_path) + ".meta.json",
                ]
                if not overwrite and all(os.path.exists(p) for p in required):
                    continue

                # Map episode to target format, then write all artifacts + sidecar
                episode = episode_map_fn(
                    episode,
                    map_step=dataset_config["step_map_fn"],
                )
                _write_episode(
                    base_path,
                    episode,
                    fps=fps,
                    overwrite=overwrite,
                )

            except Exception as e:
                print(f"Failed to process episode {gidx} in {dataset_name}-{split}: {e}")
                continue

        if num_shards == 1:
            _assemble_manifest(split_output_dir, split)
        else:
            print(f"Sharded run (shard {shard_id}/{num_shards}): sidecars written for {split}; "
                  f"run --finalize_manifest after all shards complete.")


def main(
    dataset_name: str = "bridge_v2",
    output_dir: str = "converted_datasets",
    dataset_home: str = BRIDGE_V2_PATH,
    fps: int = 10,
    max_episodes: int | None = None,
    splits: Tuple[str, ...] = ("train", "val"),
    overwrite: bool = False,
    num_shards: int = 1,
    shard_id: int = 0,
    finalize_manifest: bool = False,
) -> None:
    """
    Entry point for downloading and converting datasets.

    Args:
        dataset_name: The name of the dataset to convert, or "all".
        output_dir: Destination path for the converted .mp4 and .npz files.
        dataset_home: Path to the Bridge TFDS builder dir (contains dataset_info.json +
            *.tfrecord shards). Defaults to BRIDGE_V2_PATH.
        fps: Target framerate for the saved MP4 videos.
        max_episodes: If set, cap the GLOBAL number of episodes processed per split
            (the cap is partitioned across shards).
        splits: TFDS splits to convert
        overwrite: If True, force re-encode/rewrite existing artifacts. Default False = resumable.
        num_shards: Number of machines/shards to split each split across (default 1).
        shard_id: This process's shard index in [0, num_shards).
        finalize_manifest: Assemble manifests from existing sidecars (no TFDS work) and return;
            run once after all shards finish.
    """
    dataset_configs = get_dataset_configs(dataset_home=dataset_home)

    if dataset_name not in dataset_configs.keys():
        raise ValueError(f"Invalid dataset name: {dataset_name}.")
    selected_configs = {dataset_name: dataset_configs[dataset_name]}

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for dataset_name, dataset_config in selected_configs.items():
        convert_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            output_dir=output_dir,
            fps=fps,
            max_episodes=max_episodes,
            splits=splits,
            overwrite=overwrite,
            num_shards=num_shards,
            shard_id=shard_id,
            finalize_manifest=finalize_manifest,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="bridge_v2")
    parser.add_argument(
        "--output_dir",
        default="./bridge/processed",
    )
    parser.add_argument(
        "--dataset_home",
        default=BRIDGE_V2_PATH,
        help="Path to the Bridge TFDS builder dir containing dataset_info.json + "
             "bridge_dataset-*.tfrecord-* shards. Defaults to the in-repo BRIDGE_V2_PATH.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit episodes per split for smoke testing.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="TFDS splits to convert.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-encode/rewrite existing artifacts (ignore existing files). NOT needed "
             "for the manifest -- it is assembled from .meta.json sidecars on every run, and "
             "the converter is resumable by default. Use this to fix a stale-mp4 desync.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split each split into this many contiguous global-index blocks so K machines "
             "can run in parallel into one shared --output_dir. All other flags + the TFDS "
             "build must be identical across shards for the {i:09d} IDs to line up.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="This process's shard index in [0, num_shards).",
    )
    parser.add_argument(
        "--finalize_manifest",
        action="store_true",
        help="Do NO TFDS work -- just assemble manifest_{split}.jsonl from existing .meta.json "
             "sidecars for each split, then exit. Run once (any machine) after all shards "
             "finish, since sharded runs (--num_shards > 1) skip the shared manifest write.",
    )
    cli = parser.parse_args()
    main(
        dataset_name=cli.dataset_name,
        output_dir=cli.output_dir,
        dataset_home=cli.dataset_home,
        fps=cli.fps,
        max_episodes=cli.max_episodes,
        splits=tuple(cli.splits),
        overwrite=cli.overwrite,
        num_shards=cli.num_shards,
        shard_id=cli.shard_id,
        finalize_manifest=cli.finalize_manifest,
    )