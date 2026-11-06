"""CPU unit tests for utils/lprof.py (and lprof_report aggregation).

No CUDA needed — lprof falls back to perf_counter timestamps. Each scenario
runs in a SUBPROCESS so LPROF_* env vars are read at a fresh import, matching
production semantics (env is read once at import time). The atexit scenario
doubly relies on subprocess isolation.

Run locally (MacOS, torch-mps env is fine):
    python tests/test_lprof.py
"""

import csv
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(REPO_ROOT)


# --------------------------------------------------------------------------
# worker bodies (run in subprocesses with parent-prepared env)
# --------------------------------------------------------------------------

def w_disabled():
    from utils import lprof
    assert lprof.region('a', si=1) is lprof.region('b'), 'no-op region must be a shared singleton'
    assert lprof.timer('a') is lprof.timer('b'), 'no-op timer must be a shared singleton'
    assert lprof.enabled_for('anything') is False
    lprof.emit('x', 1.0)
    lprof.begin_sample()
    assert lprof.flush() == 0.0
    assert lprof.end_sample() == 0.0
    # loose overhead sanity: 100k disabled call sites well under a second
    t0 = time.perf_counter()
    for _ in range(100_000):
        with lprof.region('hot.loop', si=1, layer=2):
            pass
    assert time.perf_counter() - t0 < 2.0
    out_dir = os.environ['LPROF_TEST_DIR']
    assert not os.listdir(out_dir), 'disabled lprof must not create files'


def w_basic():
    from utils import lprof
    lprof.begin_sample(prompt_id='p0')
    with lprof.region('attn.csp', si=29, layer=3, k_len=100,
                      counts_sum=lprof.lazy(lambda: 42)):
        time.sleep(0.01)
    t = lprof.timer('attn.map', si=29, layer=3)
    time.sleep(0.002)
    t.mark('group')
    t.mark('coord')
    time.sleep(0.002)
    t.note(strategy='footprint')
    t.stop('band')
    lprof.emit('scale', 12.5, si=29, repeat=1, pn=[1, 2, 3])
    lprof.begin_sample(prompt_id='p1')
    with lprof.region('e2e'):
        pass
    lprof.flush()


def w_phase():
    from utils import lprof
    lprof.set_phase('warmup')
    lprof.begin_sample()                      # sample 0
    with lprof.region('step'):
        pass
    lprof.set_phase('measure')
    lprof.begin_sample()                      # sample 1
    with lprof.region('step'):
        pass
    lprof.flush()                             # single flush: stamps must hold


def w_scope():
    from utils import lprof
    assert lprof.enabled_for('attn.map')      # via 'attn.map*'
    assert lprof.enabled_for('dcsla')         # via 'dcsla.*' prefix expansion
    assert not lprof.enabled_for('attn.csp')
    with lprof.region('attn.csp', si=1):      # filtered out
        pass
    t = lprof.timer('attn.map', si=1)
    t.stop('group')
    t2 = lprof.timer('dcsla', si=1)
    t2.mark('select')
    t2.stop('fwd')
    lprof.flush()


def w_warmup_fallback():
    from utils import lprof
    for _ in range(3):                        # samples 0,1 -> warmup; 2 -> measure
        lprof.begin_sample()
        with lprof.region('step'):
            pass
    lprof.flush()


def w_atexit():
    from utils import lprof
    lprof.begin_sample()
    with lprof.region('leaky'):
        pass
    lprof.emit('e2e', 5.0)
    # no flush: the atexit hook must settle + write everything


def w_configure():
    from utils import lprof
    lprof.configure(out_dir=os.environ['LPROF_TEST_CFG_DIR'], tag='cfgtag',
                    meta={'note': 'from-configure'})
    lprof.emit('e2e', 1.0)
    lprof.flush()


WORKERS = {n[2:]: f for n, f in list(globals().items())
           if callable(f) and n.startswith('w_')}


# --------------------------------------------------------------------------
# parent-side harness
# --------------------------------------------------------------------------

def run_worker(name, env_overrides, tmp):
    env = {k: v for k, v in os.environ.items() if not k.startswith('LPROF')}
    env.update(env_overrides)
    env['LPROF_TEST_DIR'] = tmp
    r = subprocess.run([sys.executable, osp.abspath(__file__), '--worker', name],
                       env=env, cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f'worker {name} failed:\n{r.stdout}\n{r.stderr}'
    return r


def read_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def by_comp(rows, comp):
    return [r for r in rows if r['component'] == comp]


def test_disabled():
    tmp = tempfile.mkdtemp()
    try:
        run_worker('disabled', {}, tmp)  # no LPROF env at all
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_basic_rows():
    tmp = tempfile.mkdtemp()
    try:
        run_worker('basic', {'LPROF': '1', 'LPROF_DIR': tmp, 'LPROF_TAG': 't1'}, tmp)
        path = osp.join(tmp, 'lprof_t1_rank0.csv')
        rows = read_rows(path)
        assert rows and list(rows[0].keys()) == ['tag', 'rank', 'phase', 'sample_idx',
                                                 'prompt_id', 'si', 'repeat', 'layer',
                                                 'component', 'ms', 'extra']
        csp = by_comp(rows, 'attn.csp')
        assert len(csp) == 1 and csp[0]['si'] == '29' and csp[0]['layer'] == '3'
        assert csp[0]['phase'] == 'measure' and csp[0]['sample_idx'] == '0'
        assert csp[0]['prompt_id'] == 'p0'
        assert float(csp[0]['ms']) >= 9.0, 'CPU fallback must measure the sleep'
        extra = json.loads(csp[0]['extra'])
        assert extra['k_len'] == 100 and extra['counts_sum'] == 42, 'lazy ctx resolved'
        segs = {r['component'] for r in rows}
        assert {'attn.map.group', 'attn.map.coord', 'attn.map.band'} <= segs
        assert 'attn.map' not in segs, 'timers emit segment rows only, no total row'
        band = by_comp(rows, 'attn.map.band')[0]
        assert json.loads(band['extra'])['strategy'] == 'footprint', 'note() applied'
        grp = by_comp(rows, 'attn.map.group')[0]
        assert float(grp['ms']) >= 1.0
        sc = by_comp(rows, 'scale')[0]
        assert sc['ms'] == '12.5' and sc['repeat'] == '1'
        assert json.loads(sc['extra'])['pn'] == [1, 2, 3]
        e2e = by_comp(rows, 'e2e')[0]
        assert e2e['sample_idx'] == '1' and e2e['prompt_id'] == 'p1'
        assert by_comp(rows, 'lprof.flush'), 'flush cost row present'
        meta = json.load(open(osp.join(tmp, 'lprof_t1_meta.json')))
        assert meta['tag'] == 't1' and meta['env']['LPROF'] == '1'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_phase_stamping():
    tmp = tempfile.mkdtemp()
    try:
        run_worker('phase', {'LPROF': '1', 'LPROF_DIR': tmp, 'LPROF_TAG': 'ph'}, tmp)
        rows = by_comp(read_rows(osp.join(tmp, 'lprof_ph_rank0.csv')), 'step')
        assert [(r['phase'], r['sample_idx']) for r in rows] == \
            [('warmup', '0'), ('measure', '1')], 'stamps taken at record creation'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_scopes():
    tmp = tempfile.mkdtemp()
    try:
        run_worker('scope', {'LPROF': '1', 'LPROF_DIR': tmp, 'LPROF_TAG': 'sc',
                             'LPROF_SCOPES': 'attn.map*,dcsla.*'}, tmp)
        rows = read_rows(osp.join(tmp, 'lprof_sc_rank0.csv'))
        comps = {r['component'] for r in rows}
        assert 'attn.csp' not in comps
        assert {'attn.map.group', 'dcsla.select', 'dcsla.fwd'} <= comps
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_warmup_fallback():
    tmp = tempfile.mkdtemp()
    try:
        run_worker('warmup_fallback',
                   {'LPROF': '1', 'LPROF_DIR': tmp, 'LPROF_TAG': 'wf',
                    'LPROF_WARMUP_SAMPLES': '2'}, tmp)
        rows = by_comp(read_rows(osp.join(tmp, 'lprof_wf_rank0.csv')), 'step')
        assert [r['phase'] for r in rows] == ['warmup', 'warmup', 'measure']
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_atexit_fallback():
    tmp = tempfile.mkdtemp()
    try:
        run_worker('atexit', {'LPROF': '1', 'LPROF_DIR': tmp, 'LPROF_TAG': 'ax'}, tmp)
        rows = read_rows(osp.join(tmp, 'lprof_ax_rank0.csv'))
        comps = {r['component'] for r in rows}
        assert {'leaky', 'e2e'} <= comps, 'atexit must settle un-flushed records'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_configure_defaults_and_env_precedence():
    tmp = tempfile.mkdtemp()
    cfg = tempfile.mkdtemp()
    try:
        # no env DIR/TAG: configure() defaults are honoured
        run_worker('configure', {'LPROF': '1', 'LPROF_TEST_CFG_DIR': cfg}, tmp)
        assert osp.exists(osp.join(cfg, 'lprof_cfgtag_rank0.csv'))
        meta = json.load(open(osp.join(cfg, 'lprof_cfgtag_meta.json')))
        assert meta['note'] == 'from-configure'
        # env DIR/TAG set: env wins over configure()
        run_worker('configure', {'LPROF': '1', 'LPROF_DIR': tmp, 'LPROF_TAG': 'envtag',
                                 'LPROF_TEST_CFG_DIR': cfg}, tmp)
        assert osp.exists(osp.join(tmp, 'lprof_envtag_rank0.csv'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(cfg, ignore_errors=True)


def test_report_aggregation():
    """Feed a hand-built CSV through tools/lprof_report.py and check numbers."""
    tmp = tempfile.mkdtemp()
    try:
        path = osp.join(tmp, 'lprof_rep_rank0.csv')
        fields = ('tag', 'rank', 'phase', 'sample_idx', 'prompt_id', 'si',
                  'repeat', 'layer', 'component', 'ms', 'extra')
        rows = [
            ('rep', 0, 'warmup', 0, 'p', 29, 0, 1, 'dcsla.fwd', 99.0, ''),
            ('rep', 0, 'measure', 1, 'p', 29, 0, 1, 'dcsla.fwd', 20.0, ''),
            ('rep', 0, 'measure', 1, 'p', 29, 0, 2, 'dcsla.fwd', 30.0, ''),
            ('rep', 0, 'measure', 2, 'p', 29, 0, 1, 'dcsla.fwd', 26.0, ''),
            ('rep', 0, 'measure', 1, 'p', 28, 0, 1, 'attn.csp', 10.0, ''),
        ]
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(fields)
            w.writerows(rows)
        r = subprocess.run([sys.executable, osp.join(REPO_ROOT, 'tools', 'lprof_report.py'),
                            path], capture_output=True, text=True, cwd=REPO_ROOT)
        assert r.returncode == 0, r.stderr
        agg = read_rows(osp.join(tmp, 'lprof_rep_rank0_agg.csv'))
        fwd = [a for a in agg if a['component'] == 'dcsla.fwd' and a['si'] == '29']
        assert len(fwd) == 1
        row = fwd[0]
        assert row['n'] == '3', 'warmup row must be excluded by default'
        assert abs(float(row['mean_ms']) - (20 + 30 + 26) / 3.0) < 1e-3  # csv stores %.4f
        assert abs(float(row['per_sample_ms']) - 76.0 / 2) < 1e-3, '2 distinct samples'
        assert 'dcsla.fwd' in r.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [test_disabled, test_basic_rows, test_phase_stamping, test_scopes,
         test_warmup_fallback, test_atexit_fallback,
         test_configure_defaults_and_env_precedence, test_report_aggregation]


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        WORKERS[sys.argv[2]]()
        return
    for t in TESTS:
        t()
        print(f'PASS {t.__name__}')
    print(f'\nall {len(TESTS)} lprof tests passed')


if __name__ == '__main__':
    main()
