"""Unified latency profiling (lprof) for Lvar inference.

One framework replacing the previously scattered timing mechanisms (the
deferred sparse-attn profiler, IndexMapTiming, the scale-hook JSONL writer).
Wrap any code block in ``region()``/``timer()`` to make it a measurable
component; select which components are measured with env vars; every
measurement streams into one long-format CSV per rank for pandas
analysis/plotting.

Environment variables (profiling is OFF by default):

  LPROF=1                  master switch. Off => every API is a shared no-op.
  LPROF_SCOPES='a,b.*'     comma-separated fnmatch patterns selecting
                           instrumentation points by NAME (default '*').
                           A pattern 'x.*' also enables the bare point 'x'
                           (e.g. 'dcsla.*' turns on the 'dcsla' timer, whose
                           rows are dcsla.select / dcsla.build / dcsla.fwd).
  LPROF_DIR=DIR            output dir (default 'work_dir/lprof'). Env wins
                           over a runner's configure(out_dir=...) default.
  LPROF_TAG=NAME           run tag, written to every row + the file name.
  LPROF_PRINT_EVERY=N      print a running per-component summary every N
                           flushed samples (0 = off).
  LPROF_WARMUP_SAMPLES=N   fallback when the runner never calls set_phase():
                           samples 0..N-1 are stamped phase='warmup'.
  LPROF_BLOCK_SPLIT=1      per-layer attn/attn_proj/mlp hook split
                           (read by tools/scale_latency_hook_utils.py).

Component naming convention (dotted hierarchy; open-ended — wrapping a new
block with region('my.thing') is all it takes to add one):
  e2e / t0 / vae_decode / scale / blocks / block.{attn,attn_proj,mlp}
  attn.{qkv,rope,kvcache} / attn.core.{dense,decision,sparse}
  attn.map(.group/.coord/.band) / attn.colsum / attn.csp
  dcsla.{select,build,fwd} / lprof.flush

Output: <LPROF_DIR>/lprof_<tag>_rank<rank>.csv with columns
  tag,rank,phase,sample_idx,prompt_id,si,repeat,layer,component,ms,extra
plus a lprof_<tag>_meta.json sidecar written by rank 0. ``extra`` is a JSON
object holding any ctx kwargs that are not si/repeat/layer/prompt_id.

Engine: torch.cuda.Event pairs recorded on the CURRENT stream and settled in
``flush()`` at sample boundaries — the hot path never synchronizes. Without
CUDA the same API falls back to perf_counter, so it is unit-testable on CPU.
Constraints: do not wrap code that switches CUDA streams internally, and do
not place region()/timer() calls inside torch.compile'd code (Event.record
would graph-break); keep them at eager call boundaries. Single-thread use is
assumed (the inference loop); no locking is done.
"""

import atexit
import csv
import fnmatch
import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict

import torch

ENV_PREFIX = 'LPROF'  # single place to rename the whole env namespace

_TRUE = {'1', 'true', 'yes', 'on'}


def _env(suffix, default=''):
    key = f'{ENV_PREFIX}_{suffix}' if suffix else ENV_PREFIX
    return os.environ.get(key, default)


_ENABLED = _env('', '0').lower() in _TRUE
_PRINT_EVERY = int(_env('PRINT_EVERY', '0') or 0)
_WARMUP_SAMPLES = int(_env('WARMUP_SAMPLES', '0') or 0)

_CSV_FIELDS = ('tag', 'rank', 'phase', 'sample_idx', 'prompt_id', 'si',
               'repeat', 'layer', 'component', 'ms', 'extra')
_CTX_COLS = ('si', 'repeat', 'layer')  # ctx kwargs promoted to CSV columns

# How many unsettled events before warning that flush() is overdue.
_OUTSTANDING_WARN = 200_000


def _parse_scopes():
    pats = [p.strip() for p in _env('SCOPES', '*').split(',') if p.strip()]
    if not pats:
        pats = ['*']
    # 'x.*' should also enable the bare instrumentation point 'x' (its rows
    # are the children x.select/x.build/...), so add the prefix as a pattern.
    pats += [p[:-2] for p in pats if p.endswith('.*') and len(p) > 2]
    return tuple(pats)


_SCOPES = _parse_scopes()


class _State:
    def __init__(self):
        self.pending = []      # unsettled measurements (own their events)
        self.rows = []         # settled row dicts waiting for the next flush
        self.phase = 'measure'
        self.phase_explicit = False
        self.sample_idx = -1
        self.prompt_id = None
        self.tag = _env('TAG', 'untagged')
        self.out_dir = _env('DIR', '')
        self.dir_from_env = bool(self.out_dir)
        self.tag_from_env = 'TAG' in {k[len(ENV_PREFIX) + 1:]
                                      for k in os.environ if k.startswith(ENV_PREFIX + '_')}
        self.meta = {}
        self.rank = int(os.environ.get('RANK', '0'))
        self.fh = None
        self.writer = None
        self.out_path = None
        self.scope_cache = {}
        self.free_events = []
        self.outstanding = 0
        self.agg = defaultdict(lambda: [0.0, 0])  # measure-phase name -> [sum_ms, n]
        self.samples_flushed = 0
        self.warned = set()
        self.has_cuda = None


_S = _State()


def _cuda_ok():
    if _S.has_cuda is None:
        _S.has_cuda = torch.cuda.is_available()
    return _S.has_cuda


def _stamp():
    """Timestamp on the current CUDA stream (Event) or perf_counter fallback."""
    if _cuda_ok():
        ev = _S.free_events.pop() if _S.free_events else torch.cuda.Event(enable_timing=True)
        ev.record()
        _S.outstanding += 1
        if _S.outstanding > _OUTSTANDING_WARN and 'outstanding' not in _S.warned:
            _S.warned.add('outstanding')
            print(f'[lprof] WARNING: >{_OUTSTANDING_WARN} unsettled events; '
                  'call lprof.flush()/end_sample() more often')
        return ev
    return time.perf_counter()


def _elapsed_ms(a, b):
    if isinstance(a, float):
        return (b - a) * 1000.0
    return a.elapsed_time(b)


def _release_segs(segs):
    if not _cuda_ok():
        return
    evs = set()
    for _seg, a, b in segs:
        evs.add(a)
        evs.add(b)
    _S.free_events.extend(evs)
    _S.outstanding -= len(evs)


def _current_phase():
    if _S.phase_explicit:
        return _S.phase
    if _WARMUP_SAMPLES > 0 and 0 <= _S.sample_idx < _WARMUP_SAMPLES:
        return 'warmup'
    return 'measure'


class _Lazy:
    """Ctx value deferred to flush time (after the CUDA sync), e.g. .item()."""
    __slots__ = ('fn',)

    def __init__(self, fn):
        self.fn = fn


class _Pending:
    __slots__ = ('name', 'ctx', 'phase', 'sample_idx', 'prompt_id', 'segs')

    def __init__(self, name, ctx):
        self.name = name
        self.ctx = ctx
        self.phase = _current_phase()
        self.sample_idx = _S.sample_idx
        self.prompt_id = _S.prompt_id
        self.segs = []  # [(seg_name_or_None, stamp_start, stamp_end)]


class _Timer:
    """Multi-segment timer: consecutive mark()s split one flow into named
    segments; rows are '<name>.<seg>'. Only stop() enqueues the record."""
    __slots__ = ('_p', '_last')

    def __init__(self, name, ctx):
        self._p = _Pending(name, ctx)
        self._last = _stamp()

    def mark(self, seg):
        now = _stamp()
        self._p.segs.append((seg, self._last, now))
        self._last = now

    def note(self, **ctx):
        """Add/override ctx after creation (e.g. a branch decided later)."""
        self._p.ctx.update(ctx)

    def stop(self, seg=None):
        if seg is not None:
            self.mark(seg)
        _S.pending.append(self._p)


class _NullTimer:
    __slots__ = ()

    def mark(self, seg):
        pass

    def note(self, **ctx):
        pass

    def stop(self, seg=None):
        pass


_NULL_TIMER = _NullTimer()


class _Region:
    __slots__ = ('_name', '_ctx', '_t0')

    def __init__(self, name, ctx):
        self._name = name
        self._ctx = ctx

    def __enter__(self):
        self._t0 = _stamp()
        return self

    def __exit__(self, exc_type, exc, tb):
        p = _Pending(self._name, self._ctx)
        p.segs.append((None, self._t0, _stamp()))
        _S.pending.append(p)
        return False


class _NullCM:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_NULL_CM = _NullCM()


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def enabled_for(name):
    """Scope check for instrumentation point `name` (cached)."""
    if not _ENABLED:
        return False
    hit = _S.scope_cache.get(name)
    if hit is None:
        hit = any(fnmatch.fnmatchcase(name, p) for p in _SCOPES)
        _S.scope_cache[name] = hit
    return hit


def region(name, **ctx):
    """Context manager timing the wrapped block as component `name`."""
    if not _ENABLED or not enabled_for(name):
        return _NULL_CM
    return _Region(name, ctx)


def timer(name, **ctx):
    """Multi-segment timer (see _Timer). Returns a shared no-op when off."""
    if not _ENABLED or not enabled_for(name):
        return _NULL_TIMER
    return _Timer(name, ctx)


def emit(name, ms, **ctx):
    """Write an already-resolved measurement (float ms) as one row."""
    if not _ENABLED or not enabled_for(name):
        return
    _S.rows.append(_make_row(name, float(ms), ctx,
                             _current_phase(), _S.sample_idx, _S.prompt_id))


def lazy(fn):
    """Wrap a ctx value so it is computed at flush time (after the sync)."""
    return _Lazy(fn)


def set_phase(phase):
    if not _ENABLED:
        return
    _S.phase = str(phase)
    _S.phase_explicit = True


def begin_sample(sample_idx=None, prompt_id=None):
    if not _ENABLED:
        return
    _S.sample_idx = int(sample_idx) if sample_idx is not None else _S.sample_idx + 1
    _S.prompt_id = prompt_id


def end_sample(synchronize=True):
    """Flush at a sample boundary. Returns wall seconds spent flushing so the
    runner can subtract profiling bookkeeping from its wall-time summary."""
    if not _ENABLED:
        return 0.0
    dt = flush(synchronize=synchronize)
    _S.samples_flushed += 1
    if _PRINT_EVERY > 0 and _S.samples_flushed % _PRINT_EVERY == 0:
        print_summary(prefix='[lprof-running]')
    return dt


def flush(synchronize=True):
    """Settle pending event pairs into CSV rows; returns wall seconds spent.
    Pass synchronize=False right after the runner's own torch.cuda.synchronize."""
    if not _ENABLED:
        return 0.0
    t0 = time.perf_counter()
    if _S.pending and synchronize and _cuda_ok():
        torch.cuda.synchronize()
    for p in _S.pending:
        for seg, a, b in p.segs:
            comp = f'{p.name}.{seg}' if seg else p.name
            _S.rows.append(_make_row(comp, _elapsed_ms(a, b), p.ctx,
                                     p.phase, p.sample_idx, p.prompt_id))
        _release_segs(p.segs)
    _S.pending.clear()
    if _S.rows:
        w = _writer()
        for row in _S.rows:
            w.writerow(row)
        _S.rows.clear()
        _S.fh.flush()
    dt = time.perf_counter() - t0
    if _S.writer is not None:  # own cost, visible in the table, excludable
        _S.writer.writerow(_make_row('lprof.flush', dt * 1000.0, {},
                                     _current_phase(), _S.sample_idx, _S.prompt_id))
        _S.fh.flush()
    return dt


def configure(out_dir=None, tag=None, meta=None):
    """Runner defaults; explicit LPROF_DIR/LPROF_TAG env always wins."""
    if not _ENABLED:
        return
    if out_dir and not _S.dir_from_env:
        if _S.writer is not None:
            _warn_once('late-dir', '[lprof] WARNING: configure(out_dir) after first flush is ignored')
        else:
            _S.out_dir = out_dir
    if tag and not _S.tag_from_env:
        if _S.writer is not None:
            _warn_once('late-tag', '[lprof] WARNING: configure(tag) after first flush is ignored')
        else:
            _S.tag = str(tag)
    if meta:
        _S.meta.update(meta)


def print_summary(prefix='[lprof]'):
    """Per-component measure-phase running summary (grep-able via `prefix`)."""
    if not _ENABLED or not _S.agg:
        return
    print(f'{prefix} measure-phase summary (tag={_S.tag}, rank={_S.rank}):')
    width = max(len(c) for c in _S.agg)
    for comp in sorted(_S.agg):
        s, n = _S.agg[comp]
        print(f'{prefix}   {comp:<{width}}  n={n:<7d} mean={s / max(n, 1):10.3f} ms  '
              f'total={s / 1000.0:9.3f} s')


# --------------------------------------------------------------------------
# internals: row building / CSV sink
# --------------------------------------------------------------------------

def _warn_once(key, msg):
    if key not in _S.warned:
        _S.warned.add(key)
        print(msg)


def _make_row(component, ms, ctx, phase, sample_idx, prompt_id):
    row = {
        'tag': _S.tag,
        'rank': _S.rank,
        'phase': phase,
        'sample_idx': sample_idx if sample_idx >= 0 else '',
        'prompt_id': prompt_id if prompt_id is not None else '',
        'si': '', 'repeat': '', 'layer': '',
        'component': component,
        'ms': round(ms, 4),
        'extra': '',
    }
    extra = {}
    for k, v in ctx.items():
        if isinstance(v, _Lazy):
            try:
                v = v.fn()
            except Exception as e:  # a lazy ctx probe must never kill the run
                v = f'<lazy-error: {e}>'
            ctx[k] = v  # cache: multi-segment records share one ctx dict
        if k in _CTX_COLS:
            row[k] = v
        elif k == 'prompt_id':
            row['prompt_id'] = v
        else:
            extra[k] = v
    if extra:
        row['extra'] = json.dumps(extra, separators=(',', ':'), default=str)
    if phase == 'measure':
        a = _S.agg[component]
        a[0] += ms
        a[1] += 1
    return row


def _writer():
    if _S.writer is None:
        out_dir = _S.out_dir or osp.join('work_dir', 'lprof')
        os.makedirs(out_dir, exist_ok=True)
        _S.out_path = osp.join(out_dir, f'lprof_{_S.tag}_rank{_S.rank}.csv')
        _S.fh = open(_S.out_path, 'w', newline='')
        _S.writer = csv.DictWriter(_S.fh, fieldnames=_CSV_FIELDS)
        _S.writer.writeheader()
        print(f'[lprof] writing rows to {_S.out_path}')
        if _S.rank == 0:
            _write_meta(out_dir)
    return _S.writer


def _write_meta(out_dir):
    meta = {
        'tag': _S.tag,
        'argv': sys.argv,
        'env': {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIX)},
        'torch': torch.__version__,
        'cuda_device': torch.cuda.get_device_name(0) if _cuda_ok() else None,
    }
    meta.update(_S.meta)
    with open(osp.join(out_dir, f'lprof_{_S.tag}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)


def _atexit():
    try:
        flush(synchronize=_cuda_ok())
    except Exception:
        pass  # CUDA context may already be torn down at interpreter exit
    try:
        if _S.fh is not None:
            _S.fh.close()
    except Exception:
        pass


if _ENABLED:
    atexit.register(_atexit)
