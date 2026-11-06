import os
import subprocess
import sysconfig
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths
import torch


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(path):
    return os.path.join(ROOT_DIR, path)

# Read requirements from requirements.txt
with open(get_path('requirements.txt')) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.lstrip().startswith('#')]

HOPPER_GENERATION = 90
DEFAULT_GENERATION = -1

sources = {
    # These kernels are only available on Hopper. There are equivalent Triton implementations for non-Hopper generations.
    'colsum_attn': {
        'source_files': { HOPPER_GENERATION: 'csrc/attn/dense_colsum_attn.cu' }
    },
    'csp_attn': {
        'source_files': { HOPPER_GENERATION: 'csrc/attn/csp_attn.cu' }
    },
    'attn': {
        'source_files': { HOPPER_GENERATION: 'csrc/attn/dense_attn.cu' }
    },
    'csp_mlp_mm1': {
        'source_files': { HOPPER_GENERATION: 'csrc/mlp/csp_mlp_mm1.cu' }
    },
    'csp_mlp_mm2_and_scatter_add': {
        'source_files': { HOPPER_GENERATION: 'csrc/mlp/csp_mlp_mm2_and_scatter_add.cu' }
    },
    'csp_scatter_add': {
        'source_files': { HOPPER_GENERATION: 'csrc/indexed_io/scatter_add.cu' }
    },
    # These kernels are available on all generations! Not just Hopper.
    'copy_indices': {
        'source_files': { DEFAULT_GENERATION: 'csrc/indexed_io/copy_indices.cu' }
    },
    'topk_indices': {
        'source_files': { DEFAULT_GENERATION: 'csrc/indexed_io/topk_indices.cu' }
    },
    'mask_to_indices': {
        'source_files': { DEFAULT_GENERATION: 'csrc/indexed_io/mask_to_indices.cu' }
    },
}

kernels = [
    'colsum_attn',
    'csp_attn',
    'attn',
    'csp_mlp_mm1',
    'csp_mlp_mm2_and_scatter_add',
    'csp_scatter_add',
    'copy_indices',
    'topk_indices',
    'mask_to_indices',
]

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
    raise RuntimeError('The released chipmunk CUDA extension requires a visible Hopper GPU (SM90).')
target = HOPPER_GENERATION

TK_REVISION = '0c44d2c7262fdd94273a18420a5861bc96a335e3'
tk_root = os.path.realpath(get_path('submodules/ThunderKittens'))
if not os.path.exists(os.path.join(tk_root, '.git')):
    raise RuntimeError('Missing ThunderKittens Git checkout. Run bash scripts/download_thunderkittens.sh from the repository root.')
try:
    checkout_root = subprocess.check_output(
        ['git', '-C', tk_root, 'rev-parse', '--show-toplevel'], text=True
    ).strip()
    revision = subprocess.check_output(
        ['git', '-C', tk_root, 'rev-parse', 'HEAD'], text=True
    ).strip()
    if os.path.realpath(checkout_root) != tk_root or revision != TK_REVISION:
        raise RuntimeError(f'ThunderKittens must be its own checkout at {TK_REVISION}; run scripts/download_thunderkittens.sh.')
    subprocess.run(['git', '-C', tk_root, 'diff', '--quiet', 'HEAD', '--'], check=True)
except subprocess.CalledProcessError as exc:
    raise RuntimeError('ThunderKittens checkout is invalid or has tracked modifications.') from exc
for required in ('include/kittens.cuh', 'prototype/prototype.cuh'):
    if not os.path.isfile(os.path.join(tk_root, required)):
        raise FileNotFoundError(f'Missing ThunderKittens header: {required}')

python_include = sysconfig.get_path('include')
torch_include = include_paths()

# CUDA flags
cuda_flags = [
    '-DNDEBUG', '-Xcompiler=-Wno-psabi', '-Xcompiler=-fno-strict-aliasing',
    '--expt-extended-lambda', '--expt-relaxed-constexpr',
    '-forward-unknown-to-host-compiler', '--use_fast_math', '-std=c++20',
    '-O3', '-Xnvlink=--verbose', '-Xptxas=--verbose', '-lineinfo',
    '-Xptxas=--warn-on-spills',
    '-DTORCH_COMPILE',
]
cpp_flags = ['-std=c++20', '-O3', '-DDPy_LIMITED_API=0x03100000']

if target == HOPPER_GENERATION:
    cuda_flags.append('-DKITTENS_HOPPER')
    cpp_flags.append('-DKITTENS_HOPPER')

arch = f'sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}'
if arch == 'sm_90': arch = 'sm_90a'
cuda_flags.append(f'-arch={arch}')

source_files = [get_path('csrc/chipmunk.cu')]

for k in kernels:
    src_files = sources[k]['source_files']
    if target not in src_files and DEFAULT_GENERATION not in src_files:
        print(f'Warning: Target {target} not found in source files for kernel {k}. We will fallback to a Triton-based implementation.')
        continue
    if target in src_files:                       # exact match, e.g. Hopper on Hopper
        source_files.append(get_path(src_files[target]))
    elif DEFAULT_GENERATION in src_files:         # portable implementation exists
        source_files.append(get_path(src_files[DEFAULT_GENERATION]))
    else:                                         # neither variant exists → skip
        raise ValueError(f'No CUDA source for kernel {k} on target {target}')
    cpp_flags.append(f'-DTK_COMPILE_{k.replace(" ", "_").upper()}')

setup(
    name='lvar-chipmunk',
    version="1.0.0",
    packages=find_packages(where=get_path("src")),
    package_dir={"": "src"},
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            'chipmunk.cuda',
            sources=source_files,
            extra_compile_args={
                'cxx': cpp_flags,
                'nvcc': cuda_flags
            },
            include_dirs=[
                python_include,
                *torch_include,
                f'{tk_root}/include',
                f'{tk_root}/prototype',
            ],
            libraries=['cuda', 'cublas', 'cudart', 'cudadevrt'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}}      
)
