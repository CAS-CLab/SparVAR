"""Evaluate PSNR, SSIM, and LPIPS for paired image profiling outputs."""

import argparse
import json
import math
import os
import os.path as osp
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImagePair:
    key: str
    baseline_path: str
    candidate_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate low-level image metrics between baseline and accelerated "
            "Infinity profiling outputs. Images are paired by relative filename."
        )
    )
    parser.add_argument(
        "--baseline-root",
        required=True,
        help="Baseline profile root or its images/ directory.",
    )
    parser.add_argument(
        "--candidate-root",
        required=True,
        help="Candidate profile root or its images/ directory.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "Summary path. Default: "
            "<candidate-root>/evaluation_results/low_level_metrics_vs_baseline_<timestamp>.json"
        ),
    )
    parser.add_argument("--device", default="cuda", help="Metric device, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of image pairs evaluated together.")
    parser.add_argument(
        "--lpips-net-type",
        choices=("alex", "vgg", "squeeze"),
        default="vgg",
        help="Backbone used by torchmetrics LPIPS.",
    )
    parser.add_argument(
        "--collect-per-image",
        action="store_true",
        help="Include per-image metrics and metric standard deviations in the output JSON.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Evaluate shared filenames even when either root has unmatched images.",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=-1,
        help="Debug option. If positive, evaluate only the first N matched filenames.",
    )
    return parser.parse_args()


def list_images(directory: str) -> Dict[str, str]:
    images = {}
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames.sort()
        for filename in sorted(filenames):
            if osp.splitext(filename)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            path = osp.join(dirpath, filename)
            key = osp.relpath(path, directory).replace(os.sep, "/")
            images[key] = path
    return images


def resolve_images_dir(root: str) -> str:
    root = osp.abspath(root)
    if not osp.isdir(root):
        raise FileNotFoundError(f"Image root does not exist: {root}")

    nested = osp.join(root, "images")
    if osp.isdir(nested) and list_images(nested):
        return nested
    if list_images(root):
        return root
    raise FileNotFoundError(f"No images found in {root} or {nested}")


def collect_pairs(
    baseline_dir: str,
    candidate_dir: str,
    allow_partial: bool,
    limit_images: int,
) -> Tuple[List[ImagePair], Dict[str, object]]:
    baseline_images = list_images(baseline_dir)
    candidate_images = list_images(candidate_dir)
    baseline_keys = set(baseline_images)
    candidate_keys = set(candidate_images)
    shared_keys = sorted(baseline_keys & candidate_keys)
    baseline_only = sorted(baseline_keys - candidate_keys)
    candidate_only = sorted(candidate_keys - baseline_keys)

    if (baseline_only or candidate_only) and not allow_partial:
        raise ValueError(
            "Baseline/candidate image sets do not match: "
            f"baseline_total={len(baseline_images)}, candidate_total={len(candidate_images)}, "
            f"shared={len(shared_keys)}, baseline_only={len(baseline_only)}, "
            f"candidate_only={len(candidate_only)}. Pass --allow-partial to evaluate only shared files."
        )
    if limit_images > 0:
        shared_keys = shared_keys[:limit_images]
    if not shared_keys:
        raise ValueError("No same-name image pairs were found.")

    pairs = [
        ImagePair(
            key=key,
            baseline_path=baseline_images[key],
            candidate_path=candidate_images[key],
        )
        for key in shared_keys
    ]
    stats = {
        "baseline_total": len(baseline_images),
        "candidate_total": len(candidate_images),
        "shared_total": len(shared_keys),
        "baseline_only": len(baseline_only),
        "candidate_only": len(candidate_only),
        "baseline_only_examples": baseline_only[:20],
        "candidate_only_examples": candidate_only[:20],
    }
    return pairs, stats


def read_rgb_image(path: str) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def build_metric_modules(device: torch.device, lpips_net_type: str):
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
    except ImportError:
        try:
            from torchmetrics import StructuralSimilarityIndexMeasure
        except ImportError as exc:
            raise ImportError("torchmetrics is required for SSIM and LPIPS evaluation.") from exc

    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError:
        try:
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
        except ImportError as exc:
            raise ImportError(
                "LPIPS requires torchmetrics with image dependencies and the lpips package."
            ) from exc

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type=lpips_net_type,
        reduction="none",
        normalize=True,
    ).to(device)
    ssim.eval()
    lpips.eval()
    return ssim, lpips


def compute_per_image_metric(
    metric,
    candidate: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    values = metric(candidate, baseline)
    metric.reset()
    values = values.detach().reshape(-1)
    if values.numel() == candidate.shape[0]:
        return values

    # Compatibility fallback for torchmetrics versions that aggregate despite reduction="none".
    per_image = []
    for image_i in range(candidate.shape[0]):
        value = metric(candidate[image_i : image_i + 1], baseline[image_i : image_i + 1])
        metric.reset()
        per_image.append(value.detach().reshape(-1)[0])
    return torch.stack(per_image)


def mean_and_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def default_output_json(candidate_dir: str) -> str:
    report_root = osp.dirname(candidate_dir) if osp.basename(candidate_dir) == "images" else candidate_dir
    save_dir = osp.join(report_root, "evaluation_results")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return osp.join(save_dir, f"low_level_metrics_vs_baseline_{timestamp}.json")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for metrics, but torch.cuda.is_available() is false.")

    baseline_dir = resolve_images_dir(args.baseline_root)
    candidate_dir = resolve_images_dir(args.candidate_root)
    pairs, pair_stats = collect_pairs(
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        allow_partial=args.allow_partial,
        limit_images=args.limit_images,
    )
    output_json = osp.abspath(args.output_json) if args.output_json else default_output_json(candidate_dir)
    ssim_metric, lpips_metric = build_metric_modules(device, args.lpips_net_type)

    print("========== Image Low-Level Metric Evaluation ==========")
    print(f"Baseline images : {baseline_dir}")
    print(f"Candidate images: {candidate_dir}")
    print(f"Matched pairs   : {len(pairs)}")
    print(f"Device          : {device}")
    print("=======================================================")

    metric_values: Dict[str, List[float]] = {"psnr": [], "ssim": [], "lpips": []}
    per_image_records: List[Dict[str, object]] = []
    failed_pairs: List[Dict[str, str]] = []

    for batch_start in tqdm(range(0, len(pairs), args.batch_size), desc="Evaluating image pairs"):
        batch_pairs = pairs[batch_start : batch_start + args.batch_size]
        loaded_pairs = []
        baseline_images = []
        candidate_images = []

        for pair in batch_pairs:
            try:
                baseline = read_rgb_image(pair.baseline_path)
                candidate = read_rgb_image(pair.candidate_path)
                if baseline.shape != candidate.shape:
                    raise ValueError(
                        f"shape mismatch: baseline={tuple(baseline.shape)}, "
                        f"candidate={tuple(candidate.shape)}"
                    )
            except Exception as exc:
                failed_pairs.append({"key": pair.key, "error": f"{type(exc).__name__}: {exc}"})
                continue
            loaded_pairs.append(pair)
            baseline_images.append(baseline)
            candidate_images.append(candidate)

        if not loaded_pairs:
            continue

        baseline_batch = torch.stack(baseline_images).to(device, non_blocking=True).float().div_(255.0)
        candidate_batch = torch.stack(candidate_images).to(device, non_blocking=True).float().div_(255.0)
        mse = (candidate_batch - baseline_batch).pow(2).flatten(start_dim=1).mean(dim=1)
        valid_mask = mse > 0

        for local_i in torch.nonzero(~valid_mask, as_tuple=False).flatten().tolist():
            failed_pairs.append(
                {
                    "key": loaded_pairs[local_i].key,
                    "error": "candidate is pixel-identical to baseline (PSNR is infinite)",
                }
            )

        if not bool(valid_mask.any()):
            continue

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        baseline_valid = baseline_batch.index_select(0, valid_indices)
        candidate_valid = candidate_batch.index_select(0, valid_indices)
        psnr_values = -10.0 * torch.log10(mse.index_select(0, valid_indices))

        with torch.inference_mode():
            ssim_values = compute_per_image_metric(ssim_metric, candidate_valid, baseline_valid)
            lpips_values = compute_per_image_metric(lpips_metric, candidate_valid, baseline_valid)

        for value_i, local_i in enumerate(valid_indices.tolist()):
            pair = loaded_pairs[local_i]
            values = {
                "psnr": float(psnr_values[value_i].item()),
                "ssim": float(ssim_values[value_i].item()),
                "lpips": float(lpips_values[value_i].item()),
            }
            for metric_name, value in values.items():
                if not math.isfinite(value):
                    raise RuntimeError(f"Non-finite {metric_name} for image pair: {pair.key}")
                metric_values[metric_name].append(value)
            if args.collect_per_image:
                per_image_records.append({"key": pair.key, **values})

    overall_metrics: Dict[str, object] = {
        "num_pairs": len(pairs),
        "num_valid_pairs": len(metric_values["psnr"]),
        "num_failed_pairs": len(failed_pairs),
    }
    for metric_name, values in metric_values.items():
        mean, std = mean_and_std(values)
        overall_metrics[metric_name] = mean
        if args.collect_per_image:
            overall_metrics[f"{metric_name}_std"] = std

    report = {
        "timestamp": datetime.now().isoformat(),
        "baseline_root": osp.abspath(args.baseline_root),
        "candidate_root": osp.abspath(args.candidate_root),
        "baseline_images_dir": baseline_dir,
        "candidate_images_dir": candidate_dir,
        "device": str(device),
        "batch_size": args.batch_size,
        "lpips_net_type": args.lpips_net_type,
        "allow_partial": args.allow_partial,
        "pair_stats": pair_stats,
        "metric_notes": {
            "pairing": "Images are paired by relative filename under each resolved images directory.",
            "aggregation": "Each metric is computed per image, then averaged uniformly over valid pairs.",
            "psnr": "RGB images normalized to [0, 1]; pixel-identical pairs are reported as failures.",
            "ssim": "torchmetrics StructuralSimilarityIndexMeasure on RGB images in [0, 1].",
            "lpips": "torchmetrics LearnedPerceptualImagePatchSimilarity with normalize=True.",
        },
        "overall_metrics": overall_metrics,
        "failed_pairs": failed_pairs,
    }
    if args.collect_per_image:
        report["per_image_metrics"] = sorted(per_image_records, key=lambda record: record["key"])

    os.makedirs(osp.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)

    print(json.dumps(overall_metrics, ensure_ascii=False, indent=2))
    if failed_pairs:
        print(f"[WARN] {len(failed_pairs)} image pair(s) were excluded; see failed_pairs in {output_json}")
    print(f"Saved metric summary to: {output_json}")


if __name__ == "__main__":
    main()
