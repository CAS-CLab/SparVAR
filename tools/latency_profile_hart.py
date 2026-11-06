import json
import random
import statistics
from pathlib import Path

import pynvml
import torch
from torch.utils.collect_env import get_pretty_env_info
from tqdm import tqdm

from tools.hart_experiment_utils import run_hart_batch
from tools.hart_profile_driver_utils import run_profile_cli
from tools.hart_profile_utils import HartProfileImageWriter
from utils.profile_utils import (default_prompts, format_memory,
                                 get_memory_usage, get_physical_device_index)


def main(args):
    if args.batch_size < 1 or args.warmup_iter < 1 or args.profile_iter < 1:
        raise ValueError('Require batch_size >= 1, warmup_iter >= 1 and profile_iter >= 1; warmup excludes compilation.')
    print(get_pretty_env_info())
    if not torch.cuda.is_available():
        raise RuntimeError('HART latency profiling requires a CUDA device.')
    from tools.run_hart import encode_prompts, llm_system_prompt, load_hart, load_hart_tokenizer
    if not args.outdir:
        args.outdir = f'work_dir/infer_profile/{args.model_type}/latency-profile/batch-{args.batch_size}'
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda', torch.cuda.current_device())
    nvml_handle = None
    nvml_initialized = False
    try:
        try:
            pynvml.nvmlInit()
            nvml_initialized = True
            physical_device_index = get_physical_device_index()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_index)
            print(f'Using {device}; monitoring physical GPU {physical_device_index}.')
        except pynvml.NVMLError as error:
            print(f'Warning: NVIDIA-SMI memory reporting is unavailable: {error}')

        torch.cuda.empty_cache()
        initial = get_memory_usage(device, nvml_handle)
        text_tokenizer, text_model = load_hart_tokenizer(args, device)
        model, ema_model = load_hart(args, device)
        active_model = ema_model if args.use_ema else model
        image_writer = HartProfileImageWriter(
            args.outdir, args.batch_size, active_model._hart_experiment['manifest']['run_id'],
        )
        saved_image_count = 0
        torch.cuda.synchronize(device)
        post_load = get_memory_usage(device, nvml_handle)
        print('\n--- Memory Usage (After Model Load and Preparation) ---')
        print(f'CPU Memory: {format_memory(post_load[0])}')
        print(f'GPU Memory (PyTorch): {format_memory(post_load[1])}')
        print(f'GPU Memory (NVIDIA-SMI): {format_memory(post_load[2])}')
        print(f'Resident generator instances: {2 if args.use_ema else 1}')

        # Choose measured prompts before warmup using an independent Python RNG.
        prompt_rng = random.Random(args.seed)
        sample_prompts = prompt_rng.sample if args.batch_size <= len(default_prompts) else prompt_rng.choices
        measured_prompts = [sample_prompts(default_prompts, k=args.batch_size)
                            for _ in range(args.profile_iter)]
        warmup_prompts = measured_prompts[0]
        inference_options = dict(
            cfg=args.cfg, max_token_length=args.max_token_length,
            use_llm_system_prompt=args.use_llm_system_prompt,
            more_smooth=args.more_smooth, final_stage=args.final_stage,
            num_maskgit_iters=args.num_maskgit_iters,
        )
        torch.cuda.reset_peak_memory_stats(device)
        print(f'\nStarting GPU warm-up for {args.warmup_iter} iterations...')
        for iteration in tqdm(range(args.warmup_iter)):
            images, _ = run_hart_batch(
                active_model, text_model, text_tokenizer, warmup_prompts,
                encode_prompts, llm_system_prompt, g_seed=args.seed,
                record=False, **inference_options,
            )
            del images
        torch.cuda.synchronize(device)
        warmup_peak = torch.cuda.max_memory_allocated(device) / 1024**2
        torch.cuda.reset_peak_memory_stats(device)
        before_profile = get_memory_usage(device, nvml_handle)
        peak_cpu, peak_nvsmi = before_profile[0], before_profile[2]

        print(f'Starting inference speed test for {args.profile_iter} iterations...')
        timings = []
        for iteration, prompts in enumerate(tqdm(measured_prompts)):
            images, timing = run_hart_batch(
                active_model, text_model, text_tokenizer, prompts,
                encode_prompts, llm_system_prompt, g_seed=args.seed + iteration,
                sample_metadata={'phase': 'profile', 'iteration': iteration},
                **inference_options,
            )
            timings.append(timing)
            if not torch.isfinite(images.detach().cpu()).all():
                raise FloatingPointError(f'Nonfinite HART output at profile iteration {iteration}')
            # run_hart_batch has already stopped both synchronized timers.
            saved_paths = image_writer.save(images, prompts, args.seed + iteration, iteration)
            saved_image_count += len(saved_paths)
            for path in saved_paths:
                print(f'Image saved as {path.resolve()}')
            del images
            cpu, _, nvsmi = get_memory_usage(device, nvml_handle)
            peak_cpu, peak_nvsmi = max(peak_cpu, cpu), max(peak_nvsmi, nvsmi)

        batch_latencies = [item['generator_seconds'] for item in timings]
        avg_batch = statistics.mean(batch_latencies)
        std_batch = statistics.stdev(batch_latencies) if len(batch_latencies) > 1 else 0.0
        total_batch = statistics.mean(item['total_seconds'] for item in timings)
        text_batch = statistics.mean(item['text_seconds'] for item in timings)
        peak_gpu = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f'\nGeneration with batch_size = {args.batch_size} takes {total_batch:.6f}s per step (w/ text encoder).')
        print('\n--- Inference Performance ---')
        print(f'Batch Size: {args.batch_size}')
        print(f'Average Latency: {avg_batch / args.batch_size * 1000:.2f} ms')
        print(f'Latency StdDev: {std_batch / args.batch_size * 1000:.2f} ms')
        print(f'Generator Batch Latency (AR + diffusion + VAE): {avg_batch * 1000:.2f} ms')
        print(f'Text Encoder Batch Latency: {text_batch * 1000:.2f} ms')
        print(f'Throughput: {args.batch_size / avg_batch:.2f} samples/sec')
        print('\n--- Memory Usage ---')
        print(f'Warmup Peak GPU Memory (PyTorch): {format_memory(warmup_peak)}')
        print(f'Peak CPU Memory (sampled during measurement): {format_memory(peak_cpu)}')
        print(f'Peak GPU Memory (PyTorch, measured): {format_memory(peak_gpu)}')
        print(f'Peak GPU Memory (NVIDIA-SMI, sampled): {format_memory(peak_nvsmi)}')

        result = {
            'run_id': active_model._hart_experiment['manifest']['run_id'],
            'batch_size': args.batch_size,
            'warmup_iter': args.warmup_iter,
            'profile_iter': args.profile_iter,
            'generator_batch_seconds': avg_batch,
            'generator_batch_std_seconds': std_batch,
            'text_batch_seconds': text_batch,
            'total_batch_seconds': total_batch,
            'per_image_seconds': avg_batch / args.batch_size,
            'throughput_images_per_second': args.batch_size / avg_batch,
            'memory_mib': {
                'initial_pytorch': initial[1],
                'post_load_pytorch': post_load[1],
                'warmup_peak_pytorch': warmup_peak,
                'before_measurement_pytorch': before_profile[1],
                'measurement_peak_pytorch': peak_gpu,
                'measurement_increment_pytorch': peak_gpu - before_profile[1],
                'sampled_peak_cpu': peak_cpu,
                'sampled_peak_nvsmi': peak_nvsmi,
            },
            'samples': timings,
            'artifacts': {
                'profile_log': getattr(args, '_profile_log_path', None),
                'images_dir': str(image_writer.image_dir.resolve()),
                'image_manifest': str(image_writer.manifest_path.resolve()),
                'saved_images': saved_image_count,
            },
        }
        result_path = Path(args.outdir) / f'hart_profile_{result["run_id"]}.json'
        result_path.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Profile results: {result_path}')
    finally:
        if nvml_initialized:
            pynvml.nvmlShutdown()


if __name__ == '__main__':
    run_profile_cli(main)
