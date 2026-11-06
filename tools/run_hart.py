import copy
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import transformers
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from tools.hart_experiment_utils import attach_hart_experiment, run_hart_batch
from utils.hart_infer_utils import get_hart_method_options, resolve_hart_model_class

# ============
# * constants
# ============
default_prompts = [
    "beautiful lady, freckles, big smile, blue eyes, short ginger hair, dark makeup, wearing a floral blue vest top, soft light, dark grey background",
    "anthropomorphic profile of the white snow owl Crystal priestess , art deco painting, pretty and expressive eyes, ornate costume, mythical, ethereal, intricate, elaborate, hyperrealism, hyper detailed, 3D, 8K, Ultra Realistic, high octane, ultra resolution, amazing detail, perfection, In frame, photorealistic, cinematic lighting, visual clarity, shading , Lumen Reflections, Super-Resolution, gigapixel, color grading, retouch, enhanced, PBR, Blender, V-ray, Procreate, zBrush, Unreal Engine 5, cinematic, volumetric, dramatic, neon lighting, wide angle lens ,no digital painting blur."
    "Bright scene, aerial view, ancient city, fantasy, gorgeous light, mirror reflection, high detail, wide angle lens.",
    "A 4k dslr image of a lemur wearing a red magician hat and a blue coat performing magic tricks with cards in a garden.",
    "A silhouette of a grand piano overlooking a dusky cityscape viewed from a top-floor penthouse, rendered in the bold and vivid sytle of a vintage travel poster.",
    "Crocodile in a sweater.",
    "Luffy from ONEPIECE, handsome face, fantasy.",
    "3d digital art of an adorable ghost, glowing within, holding a heart shaped pumpkin, Halloween, super cute, spooky haunted house background.",
    "an astronaut sitting in a diner, eating fries, cinematic, analog film",
    "Chinese architecture, ancient style,mountain, bird, lotus, pond, big tree, 4K Unity, octane rendering.",
]

llm_system_prompt = """Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation.

Examples:
- User Prompt: A cat sleeping -> A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.
- User Prompt: A busy city street -> A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.

Please generate only the enhanced description for the prompt below and DO NOT include any additional sentences. Start your response with "Enhanced Prompt:".

User Prompt:\n"""

max_seq_len = 10240
max_batch_size = 16


# ===============
# * safety check
# ===============
safety_policy = """
    The prompt must not seek to generate harmful, abusive, hateful, sexually explicit, or otherwise inappropriate content targeting individuals or protected groups.
"""


def is_dangerous(tokenizer, model, user_prompt, threshold=0.5):
    prompt = f"""You are a policy expert trying to help determine whether a user
    prompt is in violation of the defined safety policies.

    <start_of_turn>
    Human Question: {user_prompt.strip()}
    <end_of_turn>

    Our safety principle is defined in the below:

    {safety_policy.strip()}

    Does the human question violate the above principle? Your answer must start
    with 'Yes' or 'No'. And then walk through step by step to be sure we answer
    correctly.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits

    # Extract the logits for the Yes and No tokens
    vocab = tokenizer.get_vocab()
    selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]

    # Convert these logits to a probability with softmax
    probabilities = F.softmax(selected_logits, dim=0)

    # Return probability of 'Yes'
    score = probabilities[0].item()

    return score > threshold


# Modified from VILA
def tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
    padding_mode: str = "longest",
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=padding_mode,
            max_length=max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    return input_ids


def encode_prompts(
    prompts,
    text_model,
    text_tokenizer,
    text_tokenizer_max_length,
    system_prompt=None,
    use_llm_system_prompt=False,
):
    device = text_model.device
    tokenized_prompts = tokenize_fn(
        prompts,
        tokenizer=text_tokenizer,
        max_length=text_tokenizer_max_length,
        padding_mode="max_length",
    )
    context_tokens = torch.stack(tokenized_prompts).to(device)
    context_mask = context_tokens != text_tokenizer.pad_token_id
    context_position_ids = torch.cumsum(context_mask, dim=1) - 1

    if not use_llm_system_prompt:
        context_tensor = text_model(
            context_tokens, attention_mask=context_mask, output_hidden_states=True
        ).hidden_states[-1]
    else:
        system_prompt_tokens = tokenize_fn(
            [system_prompt],
            tokenizer=text_tokenizer,
            max_length=text_tokenizer_max_length,
            padding_mode="longest",
        )
        system_prompt_tokens = system_prompt_tokens[0].to(context_tokens.device)
        system_prompt_tokens = system_prompt_tokens.unsqueeze(0)
        system_prompt_tokens = system_prompt_tokens.repeat(context_tokens.shape[0], 1)
        system_prompt_mask = torch.ones_like(context_mask)[
            :, : system_prompt_tokens.shape[1]
        ]
        # include system prompt when calculating embeddings
        # but only keep the embedding for original tokens
        context_tensor = text_model(
            torch.cat([system_prompt_tokens, context_tokens], dim=1),
            attention_mask=torch.cat(
                [
                    system_prompt_mask,
                    context_mask,
                ],
                dim=1,
            ),
            output_hidden_states=True,
        ).hidden_states[-1][:, system_prompt_tokens.shape[1] :]
    context_tensor = context_tensor.float()

    return (context_tokens, context_mask, context_position_ids, context_tensor)


def save_images(sample_imgs, sample_folder_dir, store_separately, prompts):
    if not store_separately and len(sample_imgs) > 1:
        grid = torchvision.utils.make_grid(sample_imgs, nrow=12)
        grid_np = grid.to(torch.float16).permute(1, 2, 0).mul_(255).cpu().numpy()

        os.makedirs(sample_folder_dir, exist_ok=True)
        grid_np = Image.fromarray(grid_np.astype(np.uint8))
        grid_np.save(os.path.join(sample_folder_dir, f"sample_images.png"))
        print(f"Example images are saved to {sample_folder_dir}")
    else:
        # bs, 3, r, r
        sample_imgs_np = sample_imgs.mul_(255).cpu().numpy()
        num_imgs = sample_imgs_np.shape[0]
        os.makedirs(sample_folder_dir, exist_ok=True)
        for img_idx in range(num_imgs):
            cur_img = sample_imgs_np[img_idx]
            cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
            cur_img_store = Image.fromarray(cur_img)
            cur_img_store.save(os.path.join(sample_folder_dir, f"{img_idx:06d}.png"))
            print(f"Image {img_idx} saved.")

    with open(os.path.join(sample_folder_dir, "prompt.txt"), "w") as f:
        f.write("\n".join(prompts))


def load_hart_tokenizer(args, device='cuda'):
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_ckpt)
    text_model = AutoModel.from_pretrained(args.text_encoder_ckpt).to(device)
    text_model.eval()

    return text_tokenizer, text_model


def load_hart(args, device='cuda'):
    model_class = resolve_hart_model_class(args.model_type)
    model_config = model_class.config_class.from_pretrained(args.model_path)
    if getattr(args, 'vae_path', None):
        model_config.vae_path = args.vae_path
    model = model_class.from_pretrained(args.model_path, config=model_config)
    print(f'\n[Selected {args.model_type}] '
          f'model size: {sum(p.numel() for p in model.parameters())/1e9:.2f}B')
    model = model.to(device)
    model.eval()
    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin"), map_location='cpu'),
            strict=True,
        )
        ema_model.eval()
        print('Using EMA weights; both base and EMA instances remain resident.')

    active_model = ema_model if args.use_ema else model
    method_options = get_hart_method_options(args)
    model.configure_acceleration(method_options)
    if ema_model is not None:
        ema_model.configure_acceleration(method_options)
    active_model.prepare_for_inference()
    attach_hart_experiment(active_model, args, method_options)

    return model, ema_model


def gen_one_img_hart(
    hart_test,
    use_ema,
    hart_test_ema,
    text_tokenizer,
    text_encoder,
    prompt,
    cfg: float = 4.5,
    max_token_length: int = 300,
    use_llm_system_prompt: bool = True,
    more_smooth: bool = True,
    g_seed=None,
    test_speed=False,           # evaluate benchmark average latency
    final_stage=0,
    num_maskgit_iters=1,
    **kwargs,
):
    active_model = hart_test_ema if use_ema else hart_test
    if active_model is None:
        raise ValueError('EMA inference requested without an EMA model.')
    output_imgs, timings = run_hart_batch(
        active_model, text_encoder, text_tokenizer, [prompt], encode_prompts,
        llm_system_prompt, cfg=cfg, max_token_length=max_token_length,
        use_llm_system_prompt=use_llm_system_prompt, more_smooth=more_smooth,
        g_seed=g_seed, final_stage=final_stage, num_maskgit_iters=num_maskgit_iters,
        sample_metadata=kwargs.get('sample_metadata'),
    )
    cost, hart_cost = timings['total_seconds'], timings['generator_seconds']
    print(f"cost: {cost}, hart cost={hart_cost}, text_encoder cost={cost - hart_cost}")

    sample_imgs_np = output_imgs.clone().mul_(255).cpu().numpy()
    num_imgs = sample_imgs_np.shape[0]
    assert num_imgs == 1, f'Evaluation only need 1 image, but now have {num_imgs}'
    cur_img = sample_imgs_np[0]
    cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)

    if test_speed:
        return Image.fromarray(cur_img), cost, hart_cost

    return Image.fromarray(cur_img)
