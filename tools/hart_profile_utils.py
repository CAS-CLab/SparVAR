"""Image artifacts for HART latency profiling."""

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image


class HartProfileImageWriter:
    """Save the first measured image for each prompt, separately for each batch size."""

    def __init__(self, output_dir, batch_size, run_id):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if (
            not isinstance(run_id, str) or not run_id or run_id in (".", "..")
            or "/" in run_id or "\\" in run_id or "\x00" in run_id
        ):
            raise ValueError("run_id must be a non-empty directory name")
        self.batch_size = batch_size
        dirname = "images" if batch_size == 1 else f"images_batch-{batch_size}"
        self.image_dir = Path(output_dir).expanduser() / dirname / run_id
        self.manifest_path = self.image_dir / "images.jsonl"
        self._saved_prompts = set()

    def save(self, images, prompts, seed, iteration):
        """Write RGB [0, 1] BCHW tensors after the caller's timing boundaries."""
        if (
            not torch.is_tensor(images) or images.ndim != 4
            or images.shape[0] != self.batch_size or images.shape[1] != 3
            or images.shape[2] < 1 or images.shape[3] < 1
        ):
            raise ValueError("images must have shape (batch_size, 3, height, width)")
        if isinstance(prompts, str):
            raise ValueError("prompts must contain one string per image")
        prompts = list(prompts)
        if len(prompts) != self.batch_size or not all(isinstance(prompt, str) for prompt in prompts):
            raise ValueError("prompts must contain one string per image")

        saved_paths = []
        for sample_index, prompt in enumerate(prompts):
            if prompt in self._saved_prompts:
                continue
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            image_path = self.image_dir / f"{digest}.png"
            record = {
                "prompt": prompt,
                "prompt_sha256": digest,
                "filename": image_path.name,
                "seed": seed,
                "iteration": iteration,
                "sample_index": sample_index,
                "batch_size": self.batch_size,
            }
            record_line = json.dumps(record, ensure_ascii=False) + "\n"
            # Convert on CPU so image serialization does not increase GPU peak memory.
            pixels = images[sample_index].detach().cpu().float().clamp(0, 1)
            pixels = pixels.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            self.image_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(image_path, format="PNG")
            with self.manifest_path.open("a", encoding="utf-8") as output:
                output.write(record_line)
            # A failed PNG or manifest write must remain retryable.
            self._saved_prompts.add(prompt)
            saved_paths.append(image_path)
        return saved_paths
