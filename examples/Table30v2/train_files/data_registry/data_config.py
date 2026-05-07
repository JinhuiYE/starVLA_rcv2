"""Table30v2 — per-task lerobot v2.1 datasets at
``/primus_xpfs_workspace_T04/xcy/lerobot_home/local/table30v2/<task>/``.

All 30 tasks share **14d aloha-style** schema with **min_max for all 14 dims**
(including grippers). We do NOT use binary for grippers because the raw
gripper values are physical metres in [0, 0.085], all below the default
binary_threshold=0.49 -> would map to constant 0.
"""
from starVLA.dataloader.gr00t_lerobot.data_config import AgilexData50Config
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoToTensor,
    VideoToNumpy,
    VideoColorJitter,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# --------------------------------------------------------------------------
# Per-task dataset kwargs (forwarded to FilteredLeRobotSingleDataset)
# --------------------------------------------------------------------------
# Edit this dict to override defaults for any task. Tasks not listed use
# ``TASK_DATASET_KWARGS["default"]``. ``chunk_stride`` lets you turn a chunk of
# length L (action_horizon, 50) into a span of L*stride real frames so slow
# tasks (e.g. button presses) get a longer temporal context per sample.
TASK_DATASET_KWARGS = {
    "default": dict(drop_head=1, drop_tail=30, static_eps=0.05, chunk_stride=4),

    # --- DOSW1 (W1) base — 6 tasks ---
    "put_in_pen_container":            dict(drop_head=30, drop_tail=30, static_eps=0.01, chunk_stride=2),
    "hold_the_tray_with_both_hands":   dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=3),
    "tidy_up_the_makeup_table":        dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=2),
    "place_objects_into_desk_drawer":  dict(drop_head=30, drop_tail=30, static_eps=0.01, chunk_stride=2),
    "fold_the_clothes":                dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=2),
    "stack_bowls":                     dict(drop_head=20, drop_tail=30, static_eps=0.01, chunk_stride=2),

    # --- DOSW1 (W1) additional — 4 tasks ---
    "put_the_shoes_back":   dict(drop_head=30, drop_tail=30, static_eps=0.01, chunk_stride=2),
    "untie_the_shoelaces":  dict(drop_head=30, drop_tail=30, static_eps=0.01, chunk_stride=2),
    "sweep_the_trash":      dict(drop_head=30, drop_tail=30, static_eps=0.01, chunk_stride=4),
    "tie_a_knot":           dict(drop_head=30, drop_tail=30, static_eps=0.01, chunk_stride=2),

    # --- ALOHA base — 4 tasks ---
    "stamp_positioning":         dict(drop_head=30, drop_tail=30, static_eps=0.01, chunk_stride=4),
    "wipe_the_blackboard":       dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=3),
    "put_the_books_back":        dict(drop_head=15, drop_tail=240, static_eps=0.01, chunk_stride=2),
    "scoop_with_a_small_spoon":  dict(drop_head=15, drop_tail=120, static_eps=0.01, chunk_stride=2),

    # --- ALOHA additional — 6 tasks ---
    "paint_jam":                               dict(drop_head=15, drop_tail=90, static_eps=0.01, chunk_stride=2),
    "pack_the_items":                          dict(drop_head=15, drop_tail=320, static_eps=0.01, chunk_stride=2),
    "pack_the_toothbrush_holder":              dict(drop_head=15, drop_tail=120, static_eps=0.01, chunk_stride=2),
    "wrap_with_a_soft_cloth":                  dict(drop_head=15, drop_tail=240, static_eps=0.01, chunk_stride=2),
    "lint_roller_remove_dirt":                 dict(drop_head=15, drop_tail=120, static_eps=0.01, chunk_stride=2),
    "put_the_pencil_case_into_the_schoolbag":  dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=2),

    # --- ARX5 base — 7 tasks ---
    "pick_out_the_green_blocks":  dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=2),
    "hang_the_cup":               dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=2),
    "wipe_the_table":             dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=2),
    "arrange_flowers":            dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=2),
    "press_the_button":           dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=4),
    "turn_on_the_light_switch":   dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=2),
    "water_the_flowers":          dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=3),

    # --- UR5 base — 3 tasks ---
    "shred_paper":          dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=3),
    "arrange_fruits":       dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=3),
    "item_classification":  dict(drop_head=15, drop_tail=30, static_eps=0.01, chunk_stride=3),

}


class _Table30v2BaseConfig(AgilexData50Config):
    """14d aloha-style + min_max on every key (joints AND grippers).
    Inherits modality_config (video/state/action/language keys + delta_indices)
    from AgilexData50Config; only overrides transform()."""

    def transform(self):
        return ComposedModalityTransform(transforms=[
            # video color jitter (only effective in train mode; eval is no-op)
            VideoToTensor(apply_to=self.video_keys),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state / action normalization (min_max on every dim)
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={k: "mean_std" for k in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={k: "mean_std" for k in self.action_keys},
            ),
        ])

    def make_dataset(self, dataset_name=None, **kwargs):
        from examples.Table30v2.train_files.filtered_dataset import (
            FilteredLeRobotSingleDataset,
        )
        kw = dict(TASK_DATASET_KWARGS.get(
            dataset_name, TASK_DATASET_KWARGS["default"]))
        return FilteredLeRobotSingleDataset(**kwargs, **kw)


# --------------------------------------------------------------------------
# Per-robot DataConfig subclasses
# --------------------------------------------------------------------------
# All four robots share the 14d aloha-style padded schema, transform and
# make_dataset logic in _Table30v2BaseConfig — the only per-robot difference is
# embodiment_tag (kept as a classvar so future per-robot tweaks can override
# transform()/modality_config() in just one subclass without touching the base).

class _Table30v2DOSW1Config(_Table30v2BaseConfig):
    embodiment_tag = EmbodimentTag.TABLE30V2_DOSW1


class _Table30v2AlohaConfig(_Table30v2BaseConfig):
    embodiment_tag = EmbodimentTag.TABLE30V2_ALOHA


class _Table30v2ARX5Config(_Table30v2BaseConfig):
    embodiment_tag = EmbodimentTag.TABLE30V2_ARX5


class _Table30v2UR5Config(_Table30v2BaseConfig):
    embodiment_tag = EmbodimentTag.TABLE30V2_UR5


# Override starvla base registry (which has these keys -> Table30v2DataConfig with q99)
ROBOT_TYPE_CONFIG_MAP = {
    "table30v2_dosw1": _Table30v2DOSW1Config(),
    "table30v2_aloha": _Table30v2AlohaConfig(),
    "table30v2_arx5":  _Table30v2ARX5Config(),
    "table30v2_ur5":   _Table30v2UR5Config(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "table30v2_aloha": EmbodimentTag.TABLE30V2_ALOHA,
    "table30v2_dosw1": EmbodimentTag.TABLE30V2_DOSW1,
    "table30v2_arx5":  EmbodimentTag.TABLE30V2_ARX5,
    "table30v2_ur5":   EmbodimentTag.TABLE30V2_UR5,
}


# --- task buckets ---
_W1_BASE = (
    "put_in_pen_container",
    "hold_the_tray_with_both_hands",
    "tidy_up_the_makeup_table",
    "place_objects_into_desk_drawer",
    "fold_the_clothes",
    "stack_bowls",
)
_W1_ADDITIONAL = (
    "put_the_shoes_back",
    "untie_the_shoelaces",
    "sweep_the_trash",
    "tie_a_knot",
)
_ALOHA_BASE = (
    "stamp_positioning",
    "wipe_the_blackboard",
    "put_the_books_back",
    "scoop_with_a_small_spoon",
)
_ALOHA_ADDITIONAL = (
    "paint_jam",
    "pack_the_items",
    "pack_the_toothbrush_holder",
    "wrap_with_a_soft_cloth",
    "lint_roller_remove_dirt",
    "put_the_pencil_case_into_the_schoolbag",
)
_ARX5_BASE = (
    "pick_out_the_green_blocks",
    "hang_the_cup",
    "wipe_the_table",
    "arrange_flowers",
    "press_the_button",
    "turn_on_the_light_switch",
    "water_the_flowers",
)
_UR5_BASE = (
    "shred_paper",
    "arrange_fruits",
    "item_classification",
)


# --------------------------------------------------------------------------
# Per-task sampling weights (used in DATASET_NAMED_MIXTURES via _entries).
# --------------------------------------------------------------------------
# Set >1.0 to boost a task (sampled more often), <1.0 to suppress it. Tasks
# not listed fall back to TASK_WEIGHTS["default"]. Final sampling probability
# also depends on `balance_dataset_weights` / `balance_trajectory_weights` in
# yaml (default False -> P(dataset) ∝ weight * len(dataset)).
TASK_WEIGHTS = {
    "default": 1.0,
    # --- DOSW1 (W1) base — 6 tasks ---
    "put_in_pen_container":            1.0,
    "hold_the_tray_with_both_hands":   1.0,
    "tidy_up_the_makeup_table":        1.0,
    "place_objects_into_desk_drawer":  1.0,
    "fold_the_clothes":                1.0,
    "stack_bowls":                     1.0,

    # --- DOSW1 (W1) additional — 4 tasks ---
    "put_the_shoes_back":   1.0,
    "untie_the_shoelaces":  1.0,
    "sweep_the_trash":      1.0,
    "tie_a_knot":           1.0,

    # --- ALOHA base — 4 tasks ---
    "stamp_positioning":         1.0,
    "wipe_the_blackboard":       1.0,
    "put_the_books_back":        1.0,
    "scoop_with_a_small_spoon":  1.0,

    # --- ALOHA additional — 6 tasks ---
    "paint_jam":                               1.0,
    "pack_the_items":                          1.0,
    "pack_the_toothbrush_holder":              1.0,
    "wrap_with_a_soft_cloth":                  1.0,
    "lint_roller_remove_dirt":                 1.0,
    "put_the_pencil_case_into_the_schoolbag":  1.0,

    # --- ARX5 base — 7 tasks ---
    "pick_out_the_green_blocks":  1.0,
    "hang_the_cup":               1.0,
    "wipe_the_table":             1.0,
    "arrange_flowers":            1.0,
    "press_the_button":           1.0,
    "turn_on_the_light_switch":   1.0,
    "water_the_flowers":          1.0,

    # --- UR5 base — 3 tasks ---
    "shred_paper":          1.0,
    "arrange_fruits":       1.0,
    "item_classification":  1.0,

}


def _entries(tasks, robot_type):
    return [(t, TASK_WEIGHTS.get(t, TASK_WEIGHTS["default"]), robot_type) for t in tasks]


DATASET_NAMED_MIXTURES = {
    "table30v2_w1_base":          _entries(_W1_BASE,    "table30v2_dosw1"),
    "table30v2_w1_additional":    _entries(_W1_ADDITIONAL,     "table30v2_dosw1"),
    "table30v2_aloha_base":       _entries(_ALOHA_BASE, "table30v2_aloha"),
    "table30v2_aloha_additional": _entries(_ALOHA_ADDITIONAL,  "table30v2_aloha"),
    "table30v2_arx5_base":        _entries(_ARX5_BASE,  "table30v2_arx5"),
    "table30v2_ur5_base":         _entries(_UR5_BASE,   "table30v2_ur5"),

    # ----- cotrain mixture: 4 robot bases combined (20 tasks total) -----
    "table30v2_cotrain_base": (
        _entries(_W1_BASE,    "table30v2_dosw1") +
        _entries(_ALOHA_BASE, "table30v2_aloha") +
        _entries(_ARX5_BASE,  "table30v2_arx5") +
        _entries(_UR5_BASE,   "table30v2_ur5")
    ),
}
