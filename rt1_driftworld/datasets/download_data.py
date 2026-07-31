"""
RT-1 (fractal20220817_data) -> per-episode mp4 + npz + txt (+ .meta.json / manifest) converter
for the drifting world model.

Conversion code is based on
https://github.com/world-model-eval/world-model-eval/blob/master/src/world_model_eval/download_data.py
https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb
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


def map_observation(
    to_step: Dict[str, Any],
    from_step: Dict[str, Any],
    from_image_feature_names: Tuple[str, ...] = ("image",),
    to_image_feature_names: Tuple[str, ...] = ("image",),
) -> None:
    """Copy selected image features from a raw TFDS step into the target step dict."""
    for from_feature_name, to_feature_name in zip(
        from_image_feature_names, to_image_feature_names
    ):
        to_step["observation"][to_feature_name] = from_step["observation"][
            from_feature_name
        ]


def rt_1_map_action(to_step: Dict[str, Any], from_step: Dict[str, Any]) -> None:
    """Pass the raw fractal action dict through unchanged (already in standardized RT-1 range)."""
    to_step["action"] = from_step["action"]


def _extract_instruction(steps_iter) -> str:
    """Return the episode's language instruction (constant per episode) from its first step."""
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
    """Map every step of an episode and stack into
    {'video': (T, H, W, C) uint8, 'action': (T, 10) float32, 'instruction': str}."""
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
    Standardize one step into a uniform observation and a 10-D action:
    world_vector(3), rotation_delta(3), gripper_closedness_action(1),
    base_displacement_vector(2), base_displacement_vertical_rotation(1).
    terminate_episode is intentionally excluded.
    """
    transformed_step = {}

    transformed_step["observation"] = {}
    transformed_step["action"] = {
        "gripper_closedness_action": np.zeros(1, dtype=np.float32),
        "rotation_delta": np.zeros(3, dtype=np.float32),
        "terminate_episode": np.zeros(3, dtype=np.int32),
        "world_vector": np.zeros(3, dtype=np.float32),
        "base_displacement_vertical_rotation": np.zeros(1, dtype=np.float32),
        "base_displacement_vector": np.zeros(2, dtype=np.float32),
    }

    # Apply dataset-specific mappings
    map_observation(transformed_step, step)
    map_action(transformed_step, step)

    # Concatenate action components into a single 10-D vector
    action = np.concatenate(
        [
            transformed_step["action"]["world_vector"],
            transformed_step["action"]["rotation_delta"],
            transformed_step["action"]["gripper_closedness_action"],
            transformed_step["action"]["base_displacement_vector"],
            transformed_step["action"]["base_displacement_vertical_rotation"],
        ],
        axis=0,
    )
    transformed_step["action"] = action

    return transformed_step


def get_dataset_configs(dataset_home: str) -> Dict[str, Dict[str, Any]]:
    """Dataset config for RT-1, under dataset_home/fractal20220817_data/0.1.0
    (local download_rt1.sh output, or "gs://gresearch/robotics")."""
    return {
        "rt_1": {
            "builder_dir": f"{dataset_home}/fractal20220817_data/0.1.0",
            "step_map_fn": functools.partial(
                step_map_fn,
                map_observation=map_observation,
                map_action=rt_1_map_action,
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
    Write one episode's {stem}.mp4/.npz/.txt/.meta.json and return its manifest row.

    When it encodes, checks the mp4 decodes to exactly the source frame count (libx264 can
    otherwise silently desync the mp4 from the action npz).
    """
    video = episode["video"]      # (T, H, W, C) uint8
    action = episode["action"]    # (T, 10) float32

    if action.shape[0] != video.shape[0]:
        raise RuntimeError(
            f"{base_path.name}: action len {action.shape[0]} != "
            f"video frames {video.shape[0]}"
        )

    video_sha1 = hashlib.sha1(np.ascontiguousarray(video).tobytes()).hexdigest()

    mp4_path = str(base_path) + ".mp4"
    if overwrite or not os.path.exists(mp4_path):
        iio.imwrite(mp4_path, video, fps=fps, codec="libx264")
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

    meta_path = str(base_path) + ".meta.json"
    if overwrite or not os.path.exists(meta_path):
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(row, f)
        os.replace(tmp_path, meta_path)

    return row


def convert_dataset(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    output_dir: str,
    fps: int,
    max_episodes: int | None = None,
    val_holdout_frac: float = 0.05,
    val_holdout_n: int | None = None,
    val_seed: int = 42,
    overwrite: bool = False,
) -> None:
    """
    Convert the RT-1 TFDS `train` split into per-episode artifacts under
    output_dir/<dataset_name>/{train,val}/.

    fractal has no val split, so val/ is a seeded random holdout of val_holdout_n (else
    round(val_holdout_frac * effective_total)) episodes; membership and the {i:09d} ids depend
    only on (val_seed, effective_total), so they are stable across runs/machines.
    """
    print(f"Converting {dataset_name}...")

    dataset_output_dir = Path(output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset...")
    try:
        dataset_builder = tfds.builder_from_directory(
            builder_dir=dataset_config["builder_dir"]
        )
    except Exception as e:
        print(f"Failed to build dataset {dataset_name}: {e}")
        return
    print("Dataset built successfully.")

    num_examples = int(dataset_builder.info.splits["train"].num_examples)
    effective_total = num_examples if max_episodes is None else min(num_examples, max_episodes)
    if val_holdout_n is not None:
        val_n = int(val_holdout_n)
    else:
        val_n = int(round(val_holdout_frac * effective_total))
    val_n = max(0, min(val_n, effective_total))

    rng = np.random.default_rng(val_seed)
    val_indices = set(rng.permutation(effective_total)[:val_n].tolist())
    print(
        f"Source train episodes: {num_examples} | processing: {effective_total} "
        f"| train: {effective_total - val_n} | val (random holdout, seed={val_seed}): {val_n}"
    )

    train_dir = dataset_output_dir / "train"
    val_dir = dataset_output_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)

    read_config = tfds.ReadConfig(
        interleave_cycle_length=1,
        interleave_block_length=1,
    )
    dataset = dataset_builder.as_dataset(
        split="train", shuffle_files=False, read_config=read_config
    )
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)

    train_count = 0
    val_count = 0

    print("Processing train split (with seeded random val holdout)...")
    for gidx, episode in tqdm(enumerate(dataset), desc=f"{dataset_name}"):
        if gidx >= effective_total:
            print(f"Reached effective_total={effective_total}; stopping.")
            break

        is_val = gidx in val_indices
        split_dir = val_dir if is_val else train_dir
        local_idx = val_count if is_val else train_count
        base_path = split_dir / f"{local_idx:09d}"

        if is_val:
            val_count += 1
        else:
            train_count += 1

        try:
            if not overwrite \
                    and os.path.exists(str(base_path) + ".mp4") \
                    and os.path.exists(str(base_path) + ".npz") \
                    and os.path.exists(str(base_path) + ".txt") \
                    and os.path.exists(str(base_path) + ".meta.json"):
                continue

            mapped = episode_map_fn(episode, map_step=dataset_config["step_map_fn"])
            _write_episode(base_path, mapped, fps=fps, overwrite=overwrite)
        except Exception as e:
            print(f"Failed to process episode g{gidx} ({base_path}): {e}")
            continue

    for split, dir_ in (("train", train_dir), ("val", val_dir)):
        meta_files = sorted(dir_.glob("*.meta.json"))
        rows = []
        for mf in meta_files:
            try:
                rows.append(json.loads(mf.read_text(encoding="utf-8")))
            except Exception as e:
                print(f"WARNING: skipping unreadable sidecar {mf}: {e} (will be regenerated next run)")
        if rows:
            manifest_path = dir_ / f"manifest_{split}.jsonl"
            with open(manifest_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"Wrote {len(rows)} manifest rows -> {manifest_path}")


def main(
    dataset_name: str = "rt_1",
    output_dir: str = "./rt_1/processed",
    dataset_home: str = "gs://gresearch/robotics",
    fps: int = 3,
    max_episodes: int | None = None,
    val_holdout_frac: float = 0.05,
    val_holdout_n: int | None = None,
    val_seed: int = 42,
    overwrite: bool = False,
) -> None:
    """
    Entry point.

    Args:
        dataset_name: only "rt_1" is supported.
        output_dir: episodes land under output_dir/rt_1/{train,val}/.
        dataset_home: parent of fractal20220817_data/0.1.0 (local path, or public GCS by default).
        fps: mp4 framerate metadata (read back by the loader).
        max_episodes: cap total episodes processed (smoke testing).
        val_holdout_frac: fraction of episodes routed to val/.
        val_holdout_n: exact number of episodes for val/ (overrides val_holdout_frac).
        val_seed: seed for the val holdout; keep fixed across machines/runs.
        overwrite: force rewrite of existing artifacts (default False = resumable).
    """
    dataset_configs = get_dataset_configs(dataset_home=dataset_home)
    if dataset_name not in dataset_configs:
        raise ValueError(f"Invalid dataset name: {dataset_name}. Choose from {list(dataset_configs)}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_configs[dataset_name],
        output_dir=output_dir,
        fps=fps,
        max_episodes=max_episodes,
        val_holdout_frac=val_holdout_frac,
        val_holdout_n=val_holdout_n,
        val_seed=val_seed,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="rt_1")
    parser.add_argument("--output_dir", default="./rt_1/processed")
    parser.add_argument(
        "--dataset_home",
        default="gs://gresearch/robotics",
        help="Parent of fractal20220817_data/0.1.0 (local path from download_rt1.sh, or GCS).",
    )
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit total episodes processed (smoke testing).",
    )
    parser.add_argument(
        "--val_holdout_frac",
        type=float,
        default=0.05,
        help="Fraction of episodes routed to a synthesized val/ split (seeded random holdout).",
    )
    parser.add_argument(
        "--val_holdout_n",
        type=int,
        default=None,
        help="Exact number of episodes for val/ (overrides --val_holdout_frac).",
    )
    parser.add_argument(
        "--val_seed",
        type=int,
        default=42,
        help="Seed for the random val holdout. Keep identical across machines/runs "
             "for matching train/val splits and IDs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-encode/rewrite existing artifacts (ignore existing files)."
    )
    cli = parser.parse_args()
    main(
        dataset_name=cli.dataset_name,
        output_dir=cli.output_dir,
        dataset_home=cli.dataset_home,
        fps=cli.fps,
        max_episodes=cli.max_episodes,
        val_holdout_frac=cli.val_holdout_frac,
        val_holdout_n=cli.val_holdout_n,
        val_seed=cli.val_seed,
        overwrite=cli.overwrite,
    )
