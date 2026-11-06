import os
os.environ['XFORMERS_FORCE_DISABLE_TRITON'] = '1'
import os.path as osp
import json
import argparse

import numpy as np
if __package__:
    from .local_cache_utils import load_cached_reward_models
else:
    from local_cache_utils import load_cached_reward_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_file", type=str, default="")
    parser.add_argument('--cache_dir', default=None)
    args = parser.parse_args()

    image_reward_model, clip_model = load_cached_reward_models(args.cache_dir)

    with open(args.meta_file, 'r') as f:
        meta_infos = json.load(f)
    
    average_image_reward = []
    average_clip_scores = []
    for meta in meta_infos:
        image_paths = meta['gen_image_paths']
        prompt = meta['prompt']
        image_rewards = image_reward_model.score(prompt, image_paths)
        _, clip_scores = clip_model.inference_rank(prompt, image_paths)
        average_image_reward.extend(image_rewards)
        average_clip_scores.extend(clip_scores)
        print(f'Average Image Reward of {len(meta_infos)} prompt and {len(average_image_reward)} images is {np.mean(average_image_reward):.4f}, Average CLIP Score is {np.mean(average_clip_scores):.4f}')
    print(f'Average Image Reward of {len(meta_infos)} prompt and {len(average_image_reward)} images is {np.mean(average_image_reward):.4f}, Average CLIP Score is {np.mean(average_clip_scores):.4f}')
    save_file = osp.join(osp.dirname(args.meta_file), 'image_reward_res.json')
    with open(save_file, 'w') as f:
        json.dump({
            'prompts': len(meta_infos),
            'images': len(average_image_reward),
            'average_image_reward': np.mean(average_image_reward),
            'average_clip_scores': np.mean(average_clip_scores)
        }, f)
    print(f'Save to {save_file}')
