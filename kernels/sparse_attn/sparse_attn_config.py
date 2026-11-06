import copy
import os
import yaml

BASE_CONFIG = {
    'num_model_invocations_per_inference_step': 1,
    'should_profile': False,
    'generation_index': 0,
    'steps': 13,
    # Multi-GPU currently only supported for Hunyuan
    'world_size': 1,

    'mlp': {
        'is_enabled': True,
        'is_fp8': False,

        'top_keys': 0.3,
        'random_keys': 0.05,
        'full_step_every': 10,
        'block_mask_cache': 2,
        'first_n_dense_layers': 2,

        'provider': 'triton', # either 'cuda' or 'triton'
    },
    "patchify": {
        'is_enabled': True,

        # To disable patching at any level, set that level's patch size to 1. To disable patching entirely, set all patch sizes to 1.
        "chunk_size_1": 8,
        "chunk_size_2": 4,
    },
    'attn': {
        'is_enabled': True,
        # which SparseDiffAttn path: 'image' (Infinity) or 'video' (InfinityStar)
        'model_family': 'video',
        # ------ CS4A ------
        'decision_scale': 10,                   # sparse decision scale `S`
        # Which repeat of the decision scale runs the full colsum decision.
        # None -> the scale's last repeat; an integer k -> repeat k (later repeats reuse its indices).
        'decision_repeat_idx': None,
        'speedup': 0,                           # 1: speed entry; 0: quality entry; TileLang uses the same native pair for both
        'colsum_fa2': False,                    # true: decision colsum = FA2 dense + QK-only Triton colsum; false: provider-native decision path
        'colsum_fa3': False,                    # true (Hopper only, needs flash_attn_interface): decision colsum = FA3 dense (o + row-lse) + QK^T-only Triton colsum; takes precedence over provider AND colsum_fa2 in dense_colsum_attn_s/_q
        'dense_attn_fa3': False,                # true (Hopper only): FA3 takes precedence over provider for all pre-sparse Qwen dense scales in basic_sparvar_infinitystar
        'top_keys': 0.05,
        'use_current_scale_topklen': False,
        'random_keys': 0.01,
        'local_voxels': 0,                      # Number of local voxels to use for static local attention
        'local_1d_window': 0,
        'cs4a_qgroup_map_mode': 'nearest',
        'cs4a_qgroup_interp_align': 'center',
        'cs4a_index_expand_mode': 'footprint',
        'cs4a_band_neighbor_frames': 1,         # frame_band: own frame +/- N neighbour frames
        'cs4a_compact_index_width': 'full',     # 'auto': analytic-width mask_to_indices output on footprint sparse steps (cuda provider only, saves the k_len-width int32 transient); 'full': legacy k_len-width allocation
        # ------ CS4A (image / Infinity only) ------
        'cs4a_image_kv_map': 'relative',        # 'relative': Decompose-Align-Project LUT; 'identity': keep decision coords (conference behavior)
        'cs4a_sink_scales': 5,                  # union tokens of the first N scales as attention sink (>= k - decision_scale when 'relative')
        'cs4a_image_expand_mode': 'center_union',  # 'center_union': mapped centers ∪ local ∪ sink; 'footprint': inverse parent-LUT mask expansion
        'cs4a_identity_cumulative': False,    # center_union + identity + nearest: inherit the previous sparse scale's valid indices, then union its own local band
        'cs4a_footprint_union_local': False,    # footprint: also union the cross-scale band prior (cs4a_ws, per-historical-scale widths)
        'use_o_cache': False,
        'update_o_cache_on_sparse_scale': 'none',
        'first_n_dense_layers': 2,
        # ------ CSLA ------
        'attn_sink': 5,
        'win_size': [-1,-1,-1,-1,-1,-1,-1,-1,1,1,3,5,7],
        'csla_scales': [10, 11, 12],            # image scales that run the static CSLA path (indices into the 13-scale schedule)
        # ------ shared FlexAttention (CSLA + D-CSLA) ------
        'flex_backend': 'triton',               # ['triton', 'fa4'] fa4 = torch>=2.11 FLASH (CuTe) flex backend, ~1.6x triton at equal density (H800-validated); auto-falls back to triton on failure
        'flex_block_size': 128,
        'flex_mode': 'max-autotune-no-cudagraphs',   # torch.compile mode; ~ / 'none' for default
        'flex_dynamic': True,                   # torch.compile(dynamic=...) for varying text/kv lengths
        # ------ D-CSLA (block-sparse FlexAttention path; alternative to csp_attn) ------
        'dcsla_enabled': False,
        'dcsla_select_mode': 'coverage',        # ['coverage', 'topk_ratio']
        'dcsla_tau': 0.9,                       # coverage: cover tau of the row's TOTAL mass (sink-covered mass credited)
        'dcsla_topk_ratio': 0.2,                # topk_ratio: fixed fraction of V (iso-budget vs cs4a top_keys)
        'dcsla_min_blocks': 16,                 # min selected video blocks per q-block row
        'dcsla_max_blocks_ratio': 0.5,          # cap on selected video blocks, fraction of V
        'dcsla_local_diag_blocks': 1,           # force +/-N blocks around the diagonal-aligned video position; 0=off
        # video-only priors; defaults live here so the image config (which never sets
        # them) can still take the D-CSLA path -- the init banner indexes them directly
        'dcsla_temporal_neighbor_frames': 0,    # force-keep the same spatial position in the +/-N neighbouring frames; 0=off
        'dcsla_temporal_diag_blocks': 0,        # radius (in blocks) around each temporally shifted center
        'dcsla_vcsla_frame_band_rho': -1,       # V-CSLA W-A frame band: keep min(2*rho+1, T) full contiguous frames per q-block (CS4A boundary-shift); -1=off; OR'd with the dynamic selection (static when budget is zeroed)
        'dcsla_keep_sink': True,                # True: force-keep all ctx(sink) blocks; False: sink selected per dcsla_sink_select_tau
        'dcsla_sink_select_tau': 0.0,           # keep_sink=False: >0 = sink region runs its OWN coverage selection (sink mass is exact, identity transfer); 0 = sink competes jointly with video
        'dcsla_mass_normalize': True,           # mass-conserving transfer (divide by footprint multiplicity); False = raw counts (v1 behavior)
        'dcsla_colsum_group_rows': 192,         # decision colsum query-group rows, multiple of 64 ({192,128,64}); Triton/TileLang colsum (CUDA speedup colsum is fixed at 192); ignored unless dcsla_enabled
        'dcsla_image_win_size': None,           # image only: per-historical-scale CSLA band width (same schedule shape as attn.win_size); None/[] = no cross-scale band prior
        # ------ CS4A + CSLA ------
        'bound_layer': 10,
        # 
        'full_step_every': 10,
        'full_step_schedule': None,

        'recompute_mask': True,
        'should_compress_indices': True,
        'should_keep_tail_dense': False,
        
        'provider': 'triton', # 'cuda', 'triton', or 'tilelang' (native dense/colsum/csp; explicit FA overrides retain priority)
    },
    "offloading": {
        'global_disable_offloading': False,

        'mlp.out_cache': False,
        'mlp.indices': False,
        'mlp.counts': False,
        'mlp.sparse_act_T': False,
        'mlp.blockmean_mid_cache': False,

        'attn.out_cache': True,
        'attn.indices': True,
        'attn.counts': False,
        'attn.lse_constants': False,

        'text_encoders': True,
    },
    "step_caching": {
        'is_enabled': False,
        'skip_step_schedule': set([7, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43])
    }
}

def get_kernel_config_mlp():
    # use the same block sizes for MLP across triton and CUDA implementations
    return {
        'bm': 128,
        'mbm': 128,
        'counts_multiple_of': 256,
    }

def get_kernel_config_attn():
    if GLOBAL_CONFIG['attn']['provider'] == 'triton': 
        # Triton-based FA2 uses a blocksize of 64x64 but we still use 192 for the BM for memory efficiency
        return {
            'bm': 192,
            'counts_multiple_of': 64,
            'indices_pad_to': 1,
        }
    elif GLOBAL_CONFIG['attn']['provider'] == 'cuda':
        # CUDA-based FA3 uses a blocksize of 192x~128
        return {
            'bm': 192,
            'counts_multiple_of': 112,
            'indices_pad_to': 4,
        }
    elif GLOBAL_CONFIG['attn']['provider'] == 'tilelang':
        # Start with Triton's index grouping and alignment; tune independently later.
        return {
            'bm': 192,
            'counts_multiple_of': 64,
            'indices_pad_to': 1,
        }
    else:
        raise ValueError(f"Invalid provider: {GLOBAL_CONFIG['attn']['provider']}")

GLOBAL_CONFIG = copy.deepcopy(BASE_CONFIG)

def update_global_config(config):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG.update({
        **GLOBAL_CONFIG,
        **config,
    })

import sys
import yaml
from typing import Dict, Any

def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> None:
    """Recursively update dictionary d with values from u"""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _deep_update(d[k], v)
        else:
            d[k] = v

def load_from_file(config_file: str) -> None:
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    # Update global config
    if yaml_config:
        _deep_update(GLOBAL_CONFIG, yaml_config)
        # update_global_config(yaml_config)
        print(f"SparVAR: using config file {config_file}")
        # print(yaml_config)
        print(yaml.dump(yaml_config, sort_keys=False, allow_unicode=True, default_flow_style=False))
        # Resolution-dependent override for runs that share one yml (e.g. the
        # pattern-stats runners launch 480p/decision 25 and 720p/decision 27 in
        # parallel). Applied after every load so a later load_from_file call
        # cannot clobber an in-memory override made before model init.
        env_decision = os.environ.get('SPARVAR_DECISION_SCALE')
        if env_decision is not None and isinstance(GLOBAL_CONFIG.get('attn'), dict):
            GLOBAL_CONFIG['attn']['decision_scale'] = int(env_decision)
            print(f"SparVAR: attn.decision_scale -> {int(env_decision)} (SPARVAR_DECISION_SCALE)")
        # Per-run overrides for the skip-last-2 timing sweep
        # (scripts/sparvar/vbench_eval_ours_speed_skip2_sweep.sh): drive
        # decision_repeat_idx / dcsla_enabled / colsum_fa3 / dense_attn_fa3 from
        # env so parallel 480p/720p processes can share this one yml without
        # editing it. Each key is only touched when its env var is set, so an
        # unset var keeps the yml value.
        if isinstance(GLOBAL_CONFIG.get('attn'), dict):
            _attn_cfg = GLOBAL_CONFIG['attn']
            env_repeat = os.environ.get('SPARVAR_DECISION_REPEAT_IDX')
            if env_repeat is not None:
                _r = env_repeat.strip().lower()
                _attn_cfg['decision_repeat_idx'] = None if _r in ('', 'null', 'none') else int(_r)
                print(f"SparVAR: attn.decision_repeat_idx -> {_attn_cfg['decision_repeat_idx']} (SPARVAR_DECISION_REPEAT_IDX)")
            for _env_key, _cfg_key in (
                ('SPARVAR_DCSLA_ENABLED', 'dcsla_enabled'),
                ('SPARVAR_COLSUM_FA3', 'colsum_fa3'),
                ('SPARVAR_DENSE_ATTN_FA3', 'dense_attn_fa3'),
            ):
                _val = os.environ.get(_env_key)
                if _val is not None:
                    _b = _val.strip().lower() in ('1', 'true', 'yes', 'on')
                    _attn_cfg[_cfg_key] = _b
                    print(f"SparVAR: attn.{_cfg_key} -> {_b} ({_env_key})")
