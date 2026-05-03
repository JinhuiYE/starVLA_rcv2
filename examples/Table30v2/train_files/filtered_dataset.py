"""Custom LeRobotSingleDataset with episode-level filter + per-task action chunk stride.

Used by examples/Table30v2/train_files/data_registry/data_config.py via the
``make_dataset`` factory hook in starVLA/dataloader/lerobot_datasets.py.
"""
from __future__ import annotations

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset


class FilteredLeRobotSingleDataset(LeRobotSingleDataset):
    """LeRobotSingleDataset with optional per-task filtering and stride.

    Args:
        drop_head: drop the first N frames of every episode.
        drop_tail: drop the last N frames of every episode.
        static_eps: if not None, drop a base_index whose chunk [base, base+max_off] has total displacement ||state[base+max_off]-state[base]|| < static_eps. Single-frame hold-position is preserved; only fully-static chunks dropped.
        chunk_stride: scale the action ``delta_indices`` by this factor so a chunk of
            length L spans ``L * chunk_stride`` real frames (uniform stride sampling).
    """

    def __init__(
        self,
        *args,
        drop_head: int = 0,
        drop_tail: int = 0,
        static_eps: float | None = None,
        chunk_stride: int = 1,
        **kwargs,
    ):
        self.drop_head = int(drop_head)
        self.drop_tail = int(drop_tail)
        self.static_eps = static_eps
        self.chunk_stride = int(chunk_stride)

        # Stride-scale action delta_indices BEFORE super().__init__() so that
        # stats compute (triggered inside super) sees the stride-scaled indices.
        # Without this, stats are computed on stride=1 actions but training samples
        # use stride=N actions -> normalization is mis-scaled.
        if self.chunk_stride != 1 and "modality_configs" in kwargs:
            mods = kwargs["modality_configs"]
            if "action" in mods:
                action_cfg = mods["action"]
                new_di = (np.asarray(action_cfg.delta_indices) * self.chunk_stride).tolist()
                # ModalityConfig is a Pydantic-like dataclass; construct a fresh one.
                mods["action"] = type(action_cfg)(
                    delta_indices=new_di,
                    modality_keys=action_cfg.modality_keys,
                )

        super().__init__(*args, **kwargs)
        # No-op: _delta_indices is already stride-scaled because modality_configs was.

    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        # Maximum positive offset of any action delta_index — used to keep
        # ``base_index + max_off`` inside the episode.
        max_off = max(
            (
                int(np.max(v))
                for k, v in self._delta_indices.items()
                if k.startswith("action.") and len(v) > 0
            ),
            default=0,
        )

        all_steps: list[tuple[int, int]] = []
        for tid, T in zip(self.trajectory_ids, self.trajectory_lengths):
            T = int(T)
            valid = np.ones(T, dtype=bool)
            if self.drop_head:
                valid[: self.drop_head] = False
            if self.drop_tail:
                valid[T - self.drop_tail :] = False
            if max_off:
                valid[T - max_off :] = False

            if self.static_eps is not None and max_off > 0:
                # Chunk-level static filter: a base is static iff EVERY dim
                # moves less than static_eps over the chunk window.
                # Equivalently: keep base when max_i |Δstate_i| >= static_eps.
                data = self.get_trajectory_data(tid)
                state = None
                if "observation.state" in data:
                    state = np.stack(data["observation.state"])
                if state is not None and state.shape[0] == T:
                    per_dim_abs = np.abs(state[max_off:] - state[: T - max_off])
                    max_disp = per_dim_abs.max(axis=1)  # shape (T - max_off,)
                    valid[: T - max_off] &= (max_disp >= self.static_eps)

            for i in np.where(valid)[0]:
                all_steps.append((tid, int(i)))

        return all_steps

    def _get_steps_config_key(self) -> str:
        """Override to include filter params so cache invalidates when knobs change."""
        import hashlib
        config_dict = {
            "delete_pause_frame": self.delete_pause_frame,
            "dataset_name": self.dataset_name,
            "drop_head": self.drop_head,
            "drop_tail": self.drop_tail,
            "static_eps": self.static_eps,
            "chunk_stride": self.chunk_stride,
        }
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Override parent to actually verify config_key against the cache.

        Parent's impl loads the pickle and returns it without checking config_key
        — so changes to filter knobs are silently ignored. We rebuild whenever
        the stored config_key doesn't match the current one.
        """
        import os, pickle
        import torch.distributed as dist
        import pandas as pd

        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0

        config_key = self._get_steps_config_key()
        steps_path = self.dataset_path / "meta" / "steps_data_index.pkl"

        # Try cache; bail to rebuild if config_key mismatch.
        if steps_path.exists():
            try:
                with open(steps_path, "rb") as f:
                    cached_data = pickle.load(f)
                if cached_data.get("config_key") == config_key:
                    return cached_data["steps"]
                else:
                    print(f"[steps cache] config_key mismatch for {self.dataset_name}, rebuilding")
            except Exception as e:
                print(f"[steps cache] load failed for {self.dataset_name} ({e}), rebuilding")

        if is_main():
            all_steps = self._get_all_steps_single_process()
            cache_data = {
                "config_key": config_key,
                "steps": all_steps,
                "num_trajectories": len(self.trajectory_ids),
                "total_steps": len(all_steps),
                "computed_timestamp": pd.Timestamp.now().isoformat(),
                "delete_pause_frame": self.delete_pause_frame,
                "drop_head": self.drop_head,
                "drop_tail": self.drop_tail,
                "static_eps": self.static_eps,
                "chunk_stride": self.chunk_stride,
            }
            steps_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = steps_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, steps_path)
            print(f"[steps cache] rebuilt for {self.dataset_name}: {len(all_steps)} steps")

        if dist.is_initialized():
            dist.barrier()

        with open(steps_path, "rb") as f:
            cached_data = pickle.load(f)
        return cached_data["steps"]

