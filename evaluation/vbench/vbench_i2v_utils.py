"""Helpers for the VBench-I2V (image-to-video) protocol used by infer4vbench_ddp.py.

Reference: https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v

Protocol summary (vbench2_beta_i2v):
- prompt manifest ``vbench2_i2v_full_info.json`` (vendored as ``VBench_I2V_full_info.json``):
  1118 entries with keys ``prompt_en`` / ``dimension`` / ``image_type`` / ``image_name``;
  ``VBench_I2V_full_info_fixed_seed.json`` adds a per-prompt ``seed`` (see add_fixed_seeds.py).
- reference images live in ``<VBench>/vbench2_beta_i2v/data/crop/<ratio>/<image_name>``;
  InfinityStar 720p/480p are 16:9-ish, so ``16-9`` is the ratio used by this repo.
- every prompt is sampled ``num_samples_per_prompt`` (=5) times and saved as
  ``<prompt_en>-<idx>.mp4``; unlike the T2V suite there is NO 25-sample rule for
  ``temporal_flickering`` (that dimension is not on the I2V leaderboard anyway).

This module is intentionally torch-free so it can be imported / unit-tested locally.
"""

import json
import os
import os.path as osp

# The 9 dimensions of the official VBench-I2V leaderboard, ordered as in
# TASK_INFO_I2V of cal_vbench_final_score.py (I2V-specific first, then quality).
I2V_LEADERBOARD_DIMENSIONS = [
    'i2v_subject',
    'i2v_background',
    'camera_motion',
    'subject_consistency',
    'background_consistency',
    'motion_smoothness',
    'dynamic_degree',
    'aesthetic_quality',
    'imaging_quality',
]

I2V_DEFAULT_IMAGE_RATIO = '16-9'
I2V_REQUIRED_KEYS = ('prompt_en', 'dimension', 'image_name', 'seed')


def load_vbench_i2v_items(prompt_json_path, image_root):
    """Load the VBench-I2V prompt manifest and attach the reference image path.

    Each returned item is the original dict plus ``image_path`` (absolute join of
    ``image_root`` and ``image_name``). Validation happens once, up front, so a wrong
    ``--i2v_image_root`` fails here with a readable message instead of surfacing as a
    ``cv2.imread -> None`` crash on some rank in the middle of the generation loop.
    """
    if not image_root:
        raise ValueError("VBench-I2V requires --i2v_image_root "
                         "(e.g. <VBench>/vbench2_beta_i2v/data/crop/16-9)")
    if not osp.isdir(image_root):
        raise FileNotFoundError(f"VBench-I2V image root not found: {image_root}")

    with open(prompt_json_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    missing_keys = []
    missing_images = []
    for idx, item in enumerate(items):
        absent = [k for k in I2V_REQUIRED_KEYS if k not in item]
        if absent:
            missing_keys.append((idx, absent))
            continue
        image_path = osp.join(image_root, item['image_name'])
        item['image_path'] = image_path
        if not osp.isfile(image_path):
            missing_images.append(item['image_name'])

    if missing_keys:
        head = ', '.join(f"#{idx}: {absent}" for idx, absent in missing_keys[:5])
        raise KeyError(f"{len(missing_keys)} VBench-I2V entries miss required keys "
                       f"{I2V_REQUIRED_KEYS} (did you run add_fixed_seeds.py?). First: {head}")
    if missing_images:
        unique_missing = sorted(set(missing_images))
        head = '\n  '.join(unique_missing[:10])
        raise FileNotFoundError(
            f"{len(unique_missing)} VBench-I2V reference images not found under {image_root}. "
            f"Run `sh vbench2_beta_i2v/download_data.sh` in the VBench repo. First missing:\n  {head}")
    return items


def get_num_samples(item, args):
    """Per-prompt sample count.

    T2V keeps the existing rule (25 videos for ``temporal_flickering`` prompts, see
    the VBench T2V protocol); I2V always uses ``args.num_samples_per_prompt``.
    """
    task = getattr(args, 'vbench_task', 't2v')
    if task == 't2v' and 'temporal_flickering' in item.get('dimension', []):
        return 25
    return args.num_samples_per_prompt


def is_t2v_flicker_item(item, args):
    """True when the item follows the T2V 25-sample temporal_flickering rule
    (used only for the final video-count summary in infer4vbench_ddp.py)."""
    task = getattr(args, 'vbench_task', 't2v')
    return task == 't2v' and 'temporal_flickering' in item.get('dimension', [])
