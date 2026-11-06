"""Aggregate / compare unified-profiling CSVs produced by utils/lprof.py.

Usage:
  python tools/lprof_report.py <csv-or-lprof-dir>                # aggregate
  python tools/lprof_report.py <csv-or-dir-A> <csv-or-dir-B>    # compare A vs B

Options:
  --include-warmup     keep phase=warmup rows (dropped by default)
  --rank N             which rank's file(s) to read from a dir (default 0)
  --all-ranks          read every rank file found in a dir
  --by si,repeat,layer grouping keys next to `component` (default: si,repeat)
  --view all|scale|sparse
        all    : every component (default)
        scale  : t0/scale/blocks/block.*/vae_decode/e2e pivot per (si,repeat)
                 with the legacy derived columns (overhead, attn_core, other)
        sparse : attn.map*/attn.colsum/attn.csp/dcsla.* with per-si map-share

Aggregation always also lands in `<input>_agg.csv` (long form) for pandas.
Legacy JSONL archives are served by tools/scale_latency_report.py (frozen).
"""

import argparse
import csv
import glob
import json
import os.path as osp
import statistics
import sys


def resolve_inputs(path, rank, all_ranks):
    if osp.isdir(path):
        files = sorted(glob.glob(osp.join(path, 'lprof_*_rank*.csv')))
        if not all_ranks:
            files = [f for f in files if f.endswith(f'_rank{rank}.csv')]
        if not files:
            sys.exit(f'no lprof_*_rank*.csv found in {path}')
        return files
    if not osp.exists(path):
        sys.exit(f'input not found: {path}')
    return [path]


def load(paths, include_warmup):
    rows = []
    for p in paths:
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                if not include_warmup and r.get('phase') == 'warmup':
                    continue
                try:
                    r['ms'] = float(r['ms'])
                except (ValueError, KeyError):
                    continue
                rows.append(r)
    return rows


def aggregate(rows, by_keys):
    """(component, *by_keys) -> stats dict."""
    groups = {}
    for r in rows:
        key = (r['component'],) + tuple(r.get(k, '') for k in by_keys)
        g = groups.setdefault(key, {'vals': [], 'samples': set()})
        g['vals'].append(r['ms'])
        g['samples'].add((r.get('rank', ''), r.get('sample_idx', '')))
    out = {}
    for key, g in groups.items():
        v = g['vals']
        n_samp = max(len(g['samples']), 1)
        out[key] = {
            'n': len(v),
            'mean': statistics.fmean(v),
            'std': statistics.stdev(v) if len(v) > 1 else 0.0,
            'p50': statistics.median(v),
            'max': max(v),
            'total': sum(v),
            'per_sample': sum(v) / n_samp,
            'samples': n_samp,
        }
    return out


def _si_order(si):
    if si == '':
        return (0, 0)
    if si == 't0':
        return (1, 0)
    try:
        return (2, int(si))
    except ValueError:
        return (3, 0)


def sort_keys(agg):
    return sorted(agg, key=lambda k: (k[0],) + tuple(_si_order(x) if i == 0 else str(x)
                                                     for i, x in enumerate(k[1:])))


def write_agg_csv(agg, by_keys, out_path):
    fields = ['component'] + list(by_keys) + \
        ['n', 'mean_ms', 'std_ms', 'p50_ms', 'max_ms', 'total_ms', 'per_sample_ms', 'samples']
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for key in sort_keys(agg):
            s = agg[key]
            w.writerow(list(key) +
                       [s['n'], f"{s['mean']:.4f}", f"{s['std']:.4f}", f"{s['p50']:.4f}",
                        f"{s['max']:.4f}", f"{s['total']:.4f}", f"{s['per_sample']:.4f}",
                        s['samples']])
    print(f'[lprof-report] aggregated CSV -> {out_path}')


def _fmt_key(key, by_keys):
    parts = [key[0]] + [f'{k}={v}' for k, v in zip(by_keys, key[1:]) if v != '']
    return '  '.join(parts)


def print_flat(agg, by_keys, title):
    if not agg:
        return
    print(f'\n== {title} ==')
    width = max(len(_fmt_key(k, by_keys)) for k in agg)
    print(f'{"group":<{width}}  {"n":>6} {"mean_ms":>10} {"std":>8} {"p50":>10} '
          f'{"max":>10} {"per_sample":>11}')
    for key in sort_keys(agg):
        s = agg[key]
        print(f'{_fmt_key(key, by_keys):<{width}}  {s["n"]:>6d} {s["mean"]:>10.3f} '
              f'{s["std"]:>8.3f} {s["p50"]:>10.3f} {s["max"]:>10.3f} {s["per_sample"]:>11.3f}')


SCALE_COMPONENTS = ('e2e', 't0', 'vae_decode', 'scale', 'blocks',
                    'block.attn', 'block.attn_proj', 'block.mlp')
SPARSE_PREFIXES = ('attn.map', 'attn.colsum', 'attn.csp', 'dcsla.')


def filter_view(rows, view):
    if view == 'scale':
        return [r for r in rows if r['component'] in SCALE_COMPONENTS]
    if view == 'sparse':
        return [r for r in rows
                if r['component'].startswith(('attn.map', 'dcsla.'))
                or r['component'] in ('attn.colsum', 'attn.csp')]
    return rows


def print_scale_view(agg):
    """Legacy per-(si,repeat) pivot with the derived columns of the old
    scale_latency_report.py (overhead = step - blocks, attn_core = attn -
    attn_proj, other = blocks - attn - mlp)."""
    def mean_of(comp, si, rep):
        s = agg.get((comp, si, rep))
        return s['mean'] if s else None

    step_keys = sorted({k[1:] for k in agg if k[0] == 'scale'},
                       key=lambda x: (_si_order(x[0]), str(x[1])))
    if not step_keys:
        return
    print('\n== scale view (per (si, repeat) means, ms) ==')
    header = f'{"si":>4} {"rep":>4} {"step":>10} {"blocks":>10} {"overhead":>9} ' \
             f'{"attn":>10} {"attn_core":>10} {"proj":>8} {"mlp":>10} {"other":>9}'
    print(header)
    for si, rep in step_keys:
        step = mean_of('scale', si, rep)
        blocks = mean_of('blocks', si, rep)
        attn = mean_of('block.attn', si, rep)
        proj = mean_of('block.attn_proj', si, rep)
        mlp = mean_of('block.mlp', si, rep)
        overhead = step - blocks if (step is not None and blocks is not None) else None
        core = attn - proj if (attn is not None and proj is not None) else None
        other = blocks - attn - mlp if None not in (blocks, attn, mlp) else None
        fmt = lambda x, w: f'{x:>{w}.3f}' if x is not None else ' ' * (w - 1) + '-'
        print(f'{si:>4} {rep:>4} {fmt(step, 10)} {fmt(blocks, 10)} {fmt(overhead, 9)} '
              f'{fmt(attn, 10)} {fmt(core, 10)} {fmt(proj, 8)} {fmt(mlp, 10)} {fmt(other, 9)}')
    for comp in ('t0', 'vae_decode', 'e2e'):
        for key in sort_keys({k: v for k, v in agg.items() if k[0] == comp}):
            s = agg[key]
            print(f'{comp:>19}: mean={s["mean"]:.3f} ms  n={s["n"]}')


def print_sparse_view(agg):
    """Per-si mapping-vs-attention split (successor of [sparse-profile] /
    [idxmap]): map = sum of attn.map* means; share = map / (map + csp)."""
    sis = sorted({k[1] for k in agg}, key=_si_order)
    print('\n== sparse view (per-si means, ms) ==')
    for si in sis:
        map_ms = sum(s['mean'] for k, s in agg.items()
                     if k[1] == si and k[0].startswith('attn.map'))
        csp = sum(s['mean'] for k, s in agg.items() if k[1] == si and k[0] == 'attn.csp')
        colsum = sum(s['mean'] for k, s in agg.items() if k[1] == si and k[0] == 'attn.colsum')
        d_sel = sum(s['mean'] for k, s in agg.items()
                    if k[1] == si and k[0] in ('dcsla.select', 'dcsla.build'))
        d_fwd = sum(s['mean'] for k, s in agg.items() if k[1] == si and k[0] == 'dcsla.fwd')
        bits = [f'si={si}']
        if map_ms or csp:
            share = 100.0 * map_ms / (map_ms + csp) if (map_ms + csp) > 0 else 0.0
            bits.append(f'map={map_ms:.3f} csp={csp:.3f} map_share={share:.1f}%')
        if colsum:
            bits.append(f'colsum={colsum:.3f}')
        if d_sel or d_fwd:
            share = 100.0 * d_sel / (d_sel + d_fwd) if (d_sel + d_fwd) > 0 else 0.0
            bits.append(f'dcsla sel+build={d_sel:.3f} fwd={d_fwd:.3f} build_share={share:.1f}%')
        print('  ' + '  |  '.join(bits))


def print_compare(agg_a, agg_b, by_keys, label_a, label_b):
    keys = sort_keys({**agg_a, **agg_b})
    print(f'\n== compare: A={label_a}  B={label_b} (mean ms) ==')
    width = max(len(_fmt_key(k, by_keys)) for k in keys)
    print(f'{"group":<{width}}  {"A":>10} {"B":>10} {"delta":>10} {"B/A":>7}')
    for k in keys:
        a = agg_a.get(k)
        b = agg_b.get(k)
        am = a['mean'] if a else None
        bm = b['mean'] if b else None
        d = (bm - am) if (am is not None and bm is not None) else None
        ratio = (bm / am) if (am and bm is not None) else None
        f = lambda x, w: f'{x:>{w}.3f}' if x is not None else ' ' * (w - 1) + '-'
        r = f'{ratio:>7.3f}' if ratio is not None else '      -'
        print(f'{_fmt_key(k, by_keys):<{width}}  {f(am, 10)} {f(bm, 10)} {f(d, 10)} {r}')


def agg_out_path(inp):
    if osp.isdir(inp):
        return osp.join(inp, 'lprof_agg.csv')
    return osp.splitext(inp)[0] + '_agg.csv'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', help='1 csv/dir = aggregate, 2 = compare')
    ap.add_argument('--include-warmup', action='store_true')
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--all-ranks', action='store_true')
    ap.add_argument('--by', default='si,repeat',
                    help='grouping keys next to component (default: si,repeat)')
    ap.add_argument('--view', choices=('all', 'scale', 'sparse'), default='all')
    args = ap.parse_args()
    if len(args.inputs) > 2:
        ap.error('at most two inputs (aggregate or A/B compare)')
    by_keys = tuple(k.strip() for k in args.by.split(',') if k.strip())

    aggs = []
    for inp in args.inputs:
        files = resolve_inputs(inp, args.rank, args.all_ranks)
        rows = filter_view(load(files, args.include_warmup), args.view)
        agg = aggregate(rows, by_keys)
        aggs.append(agg)
        write_agg_csv(agg, by_keys, agg_out_path(inp))

    if len(aggs) == 2:
        labels = [osp.basename(i.rstrip('/')) for i in args.inputs]
        print_compare(aggs[0], aggs[1], by_keys, labels[0], labels[1])
        return

    agg = aggs[0]
    if args.view == 'scale' and by_keys[:2] == ('si', 'repeat'):
        print_scale_view(agg)
    else:
        print_flat(agg, by_keys, f'lprof aggregate ({args.view} view)')
    if args.view == 'sparse' and by_keys and by_keys[0] == 'si':
        print_sparse_view(agg)


if __name__ == '__main__':
    main()
