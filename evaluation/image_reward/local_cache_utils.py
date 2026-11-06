"""Load the standard ImageReward and CLIP scorers from existing local caches."""

import os
from pathlib import Path


def load_cached_reward_models(cache_dir=None):
    cache = Path(cache_dir or os.environ.get('IMAGEREWARD_CACHE', '~/.cache/ImageReward')).expanduser().resolve()
    paths = {name: cache/name for name in ('ImageReward.pt', 'med_config.json', 'ViT-L-14.pt')}
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError('Missing local ImageReward assets: ' + ', '.join(missing))

    # Set these before importing the package so its tokenizer also uses cached files.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import ImageReward as RM

    print(f'Loading ImageReward and CLIP from local cache: {cache}', flush=True)
    reward = RM.load(str(paths['ImageReward.pt']), med_config=str(paths['med_config.json']))
    clip = RM.load_score('CLIP', download_root=str(cache))
    return reward, clip
