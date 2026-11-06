"""Shared reproducibility, timing and experiment records for HART inference."""

import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def seed_hart_inference(seed):
    """Seed both the model generator and HART's global NumPy/Torch RNG users."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def _checkpoint_record(path):
    path = Path(path).expanduser().resolve()
    record = {"path": str(path)}
    if path.is_file():
        stat = path.stat()
        record.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    elif path.is_dir():
        record["files"] = []
        for child in sorted(path.iterdir()):
            if child.name.startswith("TODO_") or child.suffix not in (".json", ".bin", ".safetensors", ".pt", ".pth"):
                continue
            if child.is_file():
                stat = child.stat()
                record["files"].append({
                    "name": child.name, "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                })
    return record


def hart_source_fingerprints(model_type):
    """Hash an explicit source allowlist, including uncommitted HART changes."""
    sources = [
        "models/hart/hart_transformer_t2i.py", "models/hart/basic_hart.py",
        "models/hart/configuration.py", "models/hart/hart_autoencoder.py",
        "models/hart/hart_autoencoder_with_disc.py", "models/vae/hart_hybrid_quant.py",
        "models/diffusion/diffloss.py", "models/diffusion/gaussian_diffusion.py",
        "models/helpers.py", "utils/hart_infer_utils.py", "utils/hart_runtime_utils.py",
        "utils/hart_arg_utils.py", "utils/infer_arg_utils.py", "utils/arg_util.py",
        "tools/run_hart.py", "tools/hart_experiment_utils.py", "tools/latency_profile_hart.py",
        "tools/hart_profile_utils.py", "tools/hart_profile_driver_utils.py",
        "scripts/hart/eval.sh", "evaluation/gen_eval/infer4eval_ddp.py",
        "evaluation/dpg_bench/infer4dpg_ddp.py", "evaluation/hpsv2/infer4hpsv2_ddp.py",
        "evaluation/image_reward/infer4reward_ddp.py",
    ]
    method_sources = {
        "sparsevar_hart": ["models/sparsevar/sparsevar_model_hart.py",
                           "models/sparsevar/basic_sparsevar_infinitystar.py"],
        "fastvar_hart": ["models/fastvar/fastvar_model_hart.py", "models/fastvar/fastvar_basic_hart.py",
                         "models/fastvar/fastvar_hart_utils.py"],
        "fastervar_hart": ["models/fastervar/fastervar_model_hart.py", "models/fastervar/fastervar_utils.py"],
        "sparvar_hart": ["models/sparvar/sparvar_model_hart.py", "models/sparvar/basic_sparvar_hart.py",
                         "models/sparvar/hart_csla_utils.py", "kernels/sparse_attn/triton/attn_csp.py",
                         "kernels/sparse_attn/triton/attn_colsum_only.py", "kernels/sparse_attn/sparse_attn_config.py"],
        "skipvar_hart": ["models/skipvar/skipvar_model_hart.py", "models/skipvar/basic_skipvar_hart.py",
                         "models/skipvar/skipvar_hart_utils.py"],
    }
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sources + method_sources.get(model_type, []) if (root / name).is_file()}


def attach_hart_experiment(model, args, method_options):
    """Attach run context after configuring and preparing the active model."""
    arg_values = args.as_dict() if hasattr(args, "as_dict") else vars(args)
    arg_values = {key: value for key, value in arg_values.items() if not key.startswith("_")}
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda}
    for package in ("transformers", "flash-attn", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    rank = int(os.environ.get("RANK", "0"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_rank-{rank}_pid-{os.getpid()}"
    config = getattr(model, "config", None)
    manifest = {
        "run_id": run_id,
        "model_type": args.model_type,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "args": _json_value(arg_values),
        "method_options": _json_value(method_options),
        "resolved_acceleration": _json_value(getattr(model, "acceleration_config", method_options)),
        "sparse_geometry": _json_value(getattr(model, "csla_geometry", None)),
        "cs4a_geometry": _json_value(getattr(model, "cs4a_geometry", None)),
        "cs4a_runtime_geometry": _json_value(getattr(model, "cs4a_runtime_geometry", None)),
        "accel_enabled": bool(getattr(model, "accel_enabled", False)),
        "dense_attn_backend": getattr(args, "dense_attn_backend", "flash_attn"),
        "config": _json_value(config.to_dict() if hasattr(config, "to_dict") else config),
        "checkpoint": _checkpoint_record(args.model_path),
        "ema_checkpoint": _checkpoint_record(Path(args.model_path) / "ema_model.bin") if args.use_ema else None,
        "use_ema": args.use_ema,
        "resident_generator_instances": 2 if args.use_ema else 1,
        "seed_protocol": "all_rng_per_sample_or_batch_v1",
        "dtype": "float16_autocast",
        "math_backend_flags": {
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "git_revision": revision,
        "source_sha256": hart_source_fingerprints(args.model_type),
        "versions": versions,
        "timing_scope": {
            "text_seconds": "tokenization and synchronized text encoding",
            "generator_seconds": "synchronized AR, diffusion and VAE decode",
            "total_seconds": "text plus generator; excludes RNG reset, serialization and image saving",
        },
    }
    if torch.cuda.is_available():
        manifest["gpu"] = torch.cuda.get_device_name()
    outdir = getattr(args, "outdir", "")
    record_path = None
    if outdir:
        image_output = Path(outdir).expanduser().resolve()
        # Keep records outside image trees consumed by benchmark evaluators.
        output = image_output.parent / f"{image_output.name}_hart_records"
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / f"hart_manifest_{run_id}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        record_path = output / f"hart_samples_{run_id}.jsonl"
        print(f"HART experiment manifest: {manifest_path}")
    model._hart_experiment = {"manifest": manifest, "record_path": record_path}
    return manifest


def record_hart_sample(model, prompts, seed, timings, sampling, sample_metadata=None):
    context = getattr(model, "_hart_experiment", None)
    if context is None or context["record_path"] is None:
        return
    record = {
        "run_id": context["manifest"]["run_id"],
        "prompts": list(prompts),
        "prompt_sha256": [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts],
        "seed": seed,
        "batch_size": len(prompts),
        "sampling": sampling,
        "timings": timings,
        "sample": sample_metadata or {},
        "inference": getattr(model, "last_inference_metadata", {}),
    }
    with context["record_path"].open("a") as output:
        output.write(json.dumps(_json_value(record), ensure_ascii=False) + "\n")


def run_hart_batch(
    model, text_model, text_tokenizer, prompts, encode_fn, system_prompt,
    *, cfg=4.5, max_token_length=300, use_llm_system_prompt=True,
    more_smooth=True, g_seed=None, final_stage=0, num_maskgit_iters=1,
    sample_metadata=None, record=True,
):
    """Run one batch with identical RNG and timing boundaries in every entrypoint."""
    seed_hart_inference(g_seed)
    device = torch.device(text_model.device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    options = getattr(model, "acceleration_config", None) or {}
    inference_context = torch.inference_mode if options.get("hart_inference_mode", True) else torch.no_grad
    with inference_context(), torch.autocast(
        device.type, dtype=torch.float16, enabled=device.type == "cuda", cache_enabled=True
    ):
        synchronize()
        start = time.perf_counter()
        _, context_mask, context_position_ids, context_tensor = encode_fn(
            prompts, text_model, text_tokenizer, max_token_length,
            system_prompt, use_llm_system_prompt,
        )
        synchronize()
        encoded = time.perf_counter()
        images = model.autoregressive_infer_cfg(
            B=context_tensor.size(0), label_B=context_tensor, cfg=cfg,
            g_seed=g_seed, more_smooth=more_smooth,
            context_position_ids=context_position_ids, context_mask=context_mask,
            final_stage=final_stage, num_maskgit_iters=num_maskgit_iters,
        )
        synchronize()
        finished = time.perf_counter()
    timings = {
        "text_seconds": encoded - start,
        "generator_seconds": finished - encoded,
        "total_seconds": finished - start,
    }
    if record:
        record_hart_sample(model, prompts, g_seed, timings, {
            "cfg": cfg, "more_smooth": more_smooth,
            "max_token_length": max_token_length,
            "use_llm_system_prompt": use_llm_system_prompt,
            "final_stage": final_stage, "num_maskgit_iters": num_maskgit_iters,
        }, sample_metadata)
    return images, timings
