"""
Language Table dataset, loading from https://huggingface.co/datasets/IPEC-COMMUNITY/language_table_lerobot
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

log = logging.getLogger(__name__)

RGB_VIDEO_KEY = "observation.images.rgb"


class LanguageTablePrecomputedDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        *,
        n_frames: int = 5,
        frame_skip: int = 1,
        action_dim: int = 2,
        action_scale: float = 20.0,
        input_h: int = 144,
        input_w: int = 256,
        split: str = "train",
        val_frac: float = 0.0,
        val_episodes_file: str | Path | None = None,
        video_key: str = RGB_VIDEO_KEY,
        max_retries: int = 100,
        full_video: bool = False,
    ) -> None:
        """
        Args:
            dataset_root: dataset root (contains meta/, data/, videos/).
            n_frames: Number of frames returned per sample.
            frame_skip: Temporal stride s (>=1). Frames are subsampled to every s-th frame, and
                each returned action is the SUM of the s consecutive per-step (x, y) deltas over that interval.
            action_dim: Action dims to keep from the raw parquet action (2 -> x,y).
            action_scale: Multiplier applied to the kept action dims.
            input_h, input_w: Target frame resolution
            split: "train" or "val"
            val_frac: Fraction of episodes for "val" set (0.0 -> everything is train).
                Ignored when `val_episodes_file` is set.
            val_episodes_file: Optional path to a JSON file containing
                a list of episode indices for the validation set
            video_key: Camera video key (only the overhead cam exists here).
            max_retries: On a corrupt/short episode at runtime, resample this many other indices before re-raising.
            full_video: If True, __getitem__ returns the whole episode for eval.
        """
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError(f"Unknown split: {split}")
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")

        self.dataset_root = Path(dataset_root)
        self.meta_dir = self.dataset_root / "meta"
        self.n_frames = int(n_frames)
        self.frame_skip = int(frame_skip)
        self.clip_len = self.n_frames * self.frame_skip
        self.action_dim = int(action_dim)
        self.action_scale = float(action_scale)
        self.input_h, self.input_w = int(input_h), int(input_w)
        self.split = split
        self.val_frac = float(val_frac)
        # Explicit held-out episode list (overrides the val_frac tail set when provided).
        self.val_episodes_file = str(val_episodes_file) if val_episodes_file else None
        self.val_episodes = (
            self._read_val_episodes(self.val_episodes_file) if self.val_episodes_file else None
        )
        self.video_key = str(video_key)
        self.max_retries = int(max_retries)
        self.full_video = bool(full_video)

        self.transform = transforms.Resize((self.input_h, self.input_w))

        # ---- meta/info.json: fps + chunking + path templates -------------------
        with open(self.meta_dir / "info.json", "r") as f:
            info = json.load(f)
        self.fps = info.get("fps", None)
        self.chunks_size = int(info.get("chunks_size", 1000))
        # LeRobot templates, e.g. "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        self.data_path_tmpl = info["data_path"]
        self.video_path_tmpl = info["video_path"]

        # ---- meta/episodes.jsonl: per-episode length + instruction -------------
        episodes = self._read_episodes()
        self.samples = self._prefilter(episodes)
        val_str = (
            f"file={self.val_episodes_file} ({len(self.val_episodes)} ids)"
            if self.val_episodes is not None else f"val_frac={self.val_frac}"
        )
        print(
            f"[LanguageTablePrecomputedDataset] {len(self.samples)} episodes kept "
            f"(split={self.split}, {val_str}, "
            f"res={self.input_h}x{self.input_w})"
        )

    # ----------------------------------------------------------- init helpers
    @staticmethod
    def _read_val_episodes(path: str | Path) -> set[int]:
        """
        Read episode indices in the validation set.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"val_episodes_file not found: {path}")
        with open(path, "r") as f:
            eids = json.load(f)
        val = {int(e) for e in eids}
        if not val:
            raise ValueError(f"val_episodes_file {path} parsed to an empty set of episode ids")
        return val

    def _read_episodes(self) -> list[dict]:
        """
        Read meta/episodes.jsonl -> [{episode_index, length, instruction}].
        """
        out: list[dict] = []
        with open(self.meta_dir / "episodes.jsonl", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                tasks = ep.get("tasks") or [""]
                out.append(
                    {
                        "episode_index": int(ep["episode_index"]),
                        "length": int(ep["length"]),
                        "instruction": (tasks[0] or "").strip(),
                    }
                )
        return out

    def _in_split(self, episode_index: int, n_total: int) -> bool:
        """
        Train/val membership. An explicit `val_episodes_file` (if set) takes precedence: "val" is
        exactly the listed episode indices, "train" is everything else. Otherwise fall back to the
        deterministic tail split: the last `val_frac` fraction of episodes (by index) is val.
        """
        if self.val_episodes is not None:
            is_val = episode_index in self.val_episodes
            return is_val if self.split == "val" else not is_val
        if self.val_frac <= 0.0:
            return self.split == "train"
        n_val = max(1, int(round(n_total * self.val_frac)))
        is_val = episode_index >= (n_total - n_val)
        return is_val if self.split == "val" else not is_val

    def _prefilter(self, episodes: list[dict]) -> list[dict]:
        n_total = len(episodes)
        kept, dropped = [], {"split": 0, "short": 0, "media": 0}
        for ep in episodes:
            eid = ep["episode_index"]
            if not self._in_split(eid, n_total):
                dropped["split"] += 1
                continue
            if ep["length"] < self.clip_len:
                dropped["short"] += 1
                continue
            if not self._video_path(eid).exists() or not self._parquet_path(eid).exists():
                dropped["media"] += 1
                continue
            kept.append(ep)
        if any(v for k, v in dropped.items() if k != "split"):
            log.warning(
                f"[LanguageTablePrecomputedDataset] dropped -- "
                f"short: {dropped['short']}, media: {dropped['media']}"
            )
        return kept

    # ------------------------------------------------------------- path helpers
    def _chunk(self, episode_index: int) -> int:
        return episode_index // self.chunks_size

    def _parquet_path(self, episode_index: int) -> Path:
        rel = self.data_path_tmpl.format(
            episode_chunk=self._chunk(episode_index), episode_index=episode_index
        )
        return self.dataset_root / rel

    def _video_path(self, episode_index: int) -> Path:
        rel = self.video_path_tmpl.format(
            episode_chunk=self._chunk(episode_index),
            video_key=self.video_key,
            episode_index=episode_index,
        )
        return self.dataset_root / rel

    def __len__(self) -> int:
        return len(self.samples)

    # --------------------------------------------------------------- loaders
    def _load_actions(self, episode_index: int, frame_ids) -> torch.Tensor:
        import pyarrow.parquet as pq

        table = pq.read_table(self._parquet_path(episode_index), columns=["action"])
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        s = self.frame_skip
        summed = np.stack(
            [actions[fid : fid + s, : self.action_dim].sum(axis=0) for fid in frame_ids],
            axis=0,
        )
        summed = summed * self.action_scale
        return torch.from_numpy(np.ascontiguousarray(summed)).float()

    def _load_frames(self, episode_index: int, frame_ids) -> torch.Tensor:
        """Random-access read frame_ids from the mp4 -> (F, input_h, input_w, C) in [0,1]."""
        from decord import VideoReader, cpu

        vr = VideoReader(str(self._video_path(episode_index)), ctx=cpu(0))
        n = len(vr)
        ids = [int(fid) if fid < n else n - 1 for fid in frame_ids]
        frames = torch.from_numpy(vr.get_batch(ids).asnumpy()).float() / 255.0  # (F,H,W,C)
        frames = frames.permute(0, 3, 1, 2)          # (F,C,H,W)
        frames = self.transform(frames)              # resize to (input_h, input_w)
        return frames.permute(0, 2, 3, 1).contiguous()  # (F,H,W,C)

    # --------------------------------------------------------------- getitem
    def __getitem__(self, index: int) -> dict:
        if self.full_video:
            return self._getitem_full(index)
        last_err = None
        for _ in range(max(1, self.max_retries)):
            try:
                return self._getitem_clip(index)
            except Exception as e:
                last_err = e
                eid = self.samples[index % len(self.samples)]["episode_index"]
                log.warning(f"[LanguageTablePrecomputedDataset] skipping episode {eid}: {e}")
                index = int(np.random.randint(0, len(self.samples)))
        raise last_err

    def _getitem_clip(self, index: int) -> dict:
        ep = self.samples[index % len(self.samples)]
        eid = ep["episode_index"]

        start = int(np.random.randint(0, ep["length"] - self.clip_len + 1))
        frame_ids = list(range(start, start + self.clip_len, self.frame_skip))
        assert len(frame_ids) == self.n_frames

        out = {"video": self._load_frames(eid, frame_ids)}
        out["action"] = self._load_actions(eid, frame_ids)

        assert out["video"].shape[0] == self.n_frames
        assert out["action"].shape == (self.n_frames, self.action_dim)
        return out

    def _getitem_full(self, index: int) -> dict:
        """Whole-episode variant (no crop), used only for evaluation."""
        ep = self.samples[index % len(self.samples)]
        eid = ep["episode_index"]
        usable = (ep["length"] // self.frame_skip) * self.frame_skip
        frame_ids = list(range(0, usable, self.frame_skip))

        out = {"video": self._load_frames(eid, frame_ids)}
        out["action"] = self._load_actions(eid, frame_ids)
        return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--split", default="train")
    args = p.parse_args()

    ds = LanguageTablePrecomputedDataset(
        dataset_root=args.dataset_root, split=args.split
    )
    item = ds[0]
    for k, v in item.items():
        if torch.is_tensor(v):
            print(f"{k}: {tuple(v.shape)} {v.dtype} | min {v.min():.4f} max {v.max():.4f}")
        else:
            print(f"{k}: {v!r}")
