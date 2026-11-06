"""CLI batch sweeps and console capture for HART profiling."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import traceback


class _Tee:
    def __init__(self, stream, log):
        self.stream = stream
        self.log = log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def run_profile_cli(profile_one_batch, argv=None):
    """Keep single-batch options intact and isolate each sweep in its own process."""
    from utils.infer_arg_utils import parse_hart_infer_args

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=None,
                        help='Profile each listed batch size in a separate process and output directory.')
    sweep, remaining = parser.parse_known_args(argv)
    if '-h' in remaining or '--help' in remaining:
        print(parser.format_help())
    args = parse_hart_infer_args(remaining)
    if args.batch_size < 1 or args.warmup_iter < 1 or args.profile_iter < 1:
        parser.error('Require batch_size >= 1, warmup_iter >= 1 and profile_iter >= 1.')

    if sweep.batch_sizes is not None:
        if any(batch < 1 for batch in sweep.batch_sizes):
            parser.error('--batch_sizes must contain positive integers.')
        if len(set(sweep.batch_sizes)) != len(sweep.batch_sizes):
            parser.error('--batch_sizes must not contain duplicate values.')
        if any(token == '--batch_size' or token.startswith('--batch_size=') for token in remaining):
            parser.error('Use either --batch_size or --batch_sizes, not both.')
        # Strip the shared output root while retaining every method/sampling flag.
        output_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        output_parser.add_argument('--outdir')
        _, worker_args = output_parser.parse_known_args(remaining)
        output_root = Path(args.outdir or f'work_dir/infer_profile/{args.model_type}/latency-profile').resolve()
        repository = Path(__file__).resolve().parents[1]
        for batch in sweep.batch_sizes:
            output_dir = output_root / f'batch-{batch}'
            print(f'\nProfiling HART batch_size={batch}; output directory: {output_dir}', flush=True)
            subprocess.run(
                [sys.executable, '-m', 'tools.latency_profile_hart', *worker_args,
                 '--batch_size', str(batch), '--outdir', str(output_dir)],
                cwd=repository, check=True,
            )
        return

    if not args.outdir:
        args.outdir = f'work_dir/infer_profile/{args.model_type}/latency-profile/batch-{args.batch_size}'
    output = Path(args.outdir)
    output.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log_path = output / f'hart_profile_{run_stamp}_pid-{os.getpid()}.log'
    args._profile_log_path = str(log_path.resolve())
    with log_path.open('w') as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print(f'Profile log: {log_path.resolve()}')
            try:
                profile_one_batch(args)
            except BaseException:
                traceback.print_exc(file=log)
                raise
