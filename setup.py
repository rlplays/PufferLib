# Debug command:
#    DEBUG=1 python setup.py build_ext --inplace --force
#    CUDA_VISIBLE_DEVICES=None LD_PRELOAD=$(gcc -print-file-name=libasan.so) python3.12 -m pufferlib.clean_pufferl eval --train.device cpu

import sys
import sysconfig
from setuptools import find_packages, find_namespace_packages, setup, Extension
import numpy
import os
import glob
import urllib.request
import zipfile
import tarfile
import platform
import shutil

import pybind11

from setuptools.command.build_ext import build_ext
import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
    ROCM_HOME
)


try:
    import ninja
except ImportError:
    print(
        "WARNING: The 'ninja' Python package is not installed (pip install ninja). "
        "Install it with 'pip install ninja' for faster extension builds especially with libtorch!"
    )
# build cuda extension if torch can find CUDA or HIP/ROCM in the system
# may require `uv pip install --no-build-isolation` or `python setup.py build_ext --inplace`
BUILD_CUDA_EXT = bool(CUDA_HOME or ROCM_HOME)

# Build with DEBUG=1 to enable debug symbols
DEBUG = os.getenv("DEBUG", "0") == "1"
NO_OCEAN = os.getenv("NO_OCEAN", "0") == "1"
NO_TRAIN = os.getenv("NO_TRAIN", "0") == "1"
NO_TORCH = os.getenv("NO_TORCH", "0") == "1"
NO_ASAN = os.getenv("NO_ASAN", "0") == "1"
SINGLE_THREADED = os.getenv("SINGLE_THREADED", "0") == "1"
NO_PUFFERLIB = os.getenv("NO_PUFFERLIB", "0") == "1"
SELF_PLAY = os.getenv("SELF_PLAY", "0") == "1"

print(f"------- DEBUG MODE? {DEBUG} | SELF_PLAY? {SELF_PLAY} -------------")
if SINGLE_THREADED:
    print("------- SINGLE THREADED MODE! -------------")
# Build raylib for your platform
RAYLIB_URL = 'https://github.com/raysan5/raylib/releases/download/5.5/'
RAYLIB_NAME = 'raylib-5.5_macos' if platform.system() == "Darwin" else 'raylib-5.5_linux_amd64'
RLIGHTS_URL = 'https://raw.githubusercontent.com/raysan5/raylib/refs/heads/master/examples/shaders/rlights.h'

def download_raylib(platform, ext):
    if not os.path.exists(platform):
        print(f'Downloading Raylib {platform}')
        urllib.request.urlretrieve(RAYLIB_URL + platform + ext, platform + ext)
        if ext == '.zip':
            with zipfile.ZipFile(platform + ext, 'r') as zip_ref:
                zip_ref.extractall()
        else:
            with tarfile.open(platform + ext, 'r') as tar_ref:
                tar_ref.extractall()

        os.remove(platform + ext)
        urllib.request.urlretrieve(RLIGHTS_URL, platform + '/include/rlights.h')

if not NO_OCEAN:
    download_raylib('raylib-5.5_webassembly', '.zip')
    download_raylib(RAYLIB_NAME, '.tar.gz')

BOX2D_URL = 'https://github.com/capnspacehook/box2d/releases/latest/download/'
BOX2D_NAME = 'box2d-macos-arm64' if platform.system() == "Darwin" else 'box2d-linux-amd64'

def download_box2d(platform):
    if not os.path.exists(platform):
        ext = ".tar.gz"

        print(f'Downloading Box2D {platform}')
        urllib.request.urlretrieve(BOX2D_URL + platform + ext, platform + ext)
        with tarfile.open(platform + ext, 'r') as tar_ref:
            tar_ref.extractall()

        os.remove(platform + ext)



if not NO_OCEAN:
    download_box2d('box2d-web')
    download_box2d(BOX2D_NAME)

# Shared compile args for all platforms
extra_compile_args = [
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    '-DPLATFORM_DESKTOP',
    '-DPUFFER_NATIVECPP_PYBINDINGS',
    '-std=gnu++20',
    '-fpermissive',
]

if SELF_PLAY:
    print("Building with self-play support (defining PUFFERLIB_SELFPLAY)")
    extra_compile_args += [
        '-DPUFFERLIB_SELFPLAY',
    ]

CUDA_INCLUDE = []
if CUDA_HOME:
    CUDA_INCLUDE.append(os.path.join(CUDA_HOME, "include"))
    print(f"Adding CUDA include path: {CUDA_INCLUDE[-1]}")

extra_link_args = [
    '-fwrapv'
]
cxx_args = [
    '-fdiagnostics-color=always',
    '-std=gnu++20',
    '-fpermissive',
]
nvcc_args = []

if DEBUG:
    extra_compile_args += [
        '-O0',
        '-g',
        '-fno-omit-frame-pointer',
        '-DDEBUG',
        '-DTORCH_USE_CUDA_DSA=1', # CUDA device side assertions
    ]
    extra_link_args += [
        '-g',
    ]
    cxx_args += [
        '-O0',
        '-g',
    ]
    nvcc_args += [
        '-O0',
        '-g',
    ]
    if not NO_ASAN:
      extra_compile_args += [
          '-fsanitize=address,undefined,bounds,pointer-overflow,leak',
      ]
      extra_link_args += [
          '-fsanitize=address,undefined,bounds,pointer-overflow,leak',
      ]
else:
    extra_compile_args += [
        '-O3',
        '-flto',
    ]
    extra_link_args += [
        '-O3',
    ]
    cxx_args += [
        '-O3',
    ]
    nvcc_args += [
        '-O3',
    ]
if SINGLE_THREADED:
    extra_compile_args += [
        '-DPUFFER_SINGLE_THREADED',
    ]

system = platform.system()
if system == 'Linux':
    extra_compile_args += [
        '-Wno-alloc-size-larger-than',
        '-Wno-odr', # One definition rule - C/C++ messiness
        '-Wno-attributes', # pybind11 
        '-Wno-unknown-pragmas', # Win VS vs Linux stuff
        # '-Wno-implicit-function-declaration', # Ignored, it's C++ not C, it's an error already.
        '-fmax-errors=3',
    ]
    extra_link_args += [
        '-Bsymbolic-functions',
    ]
elif system == 'Darwin':
    extra_compile_args += [
        '-Wno-error=int-conversion',
        '-Wno-error=incompatible-function-pointer-types',
        '-Wno-error=implicit-function-declaration',
    ]
    extra_link_args += [
        '-framework', 'Cocoa',
        '-framework', 'OpenGL',
        '-framework', 'IOKit',
    ]
else:
    raise ValueError(f'Unsupported system: {system}')

# Default Gym/Gymnasium/PettingZoo versions
# Gym:
# - 0.26 still has deprecation warnings and is the last version of the package
# - 0.25 adds a breaking API change to reset, step, and render_modes
# - 0.24 is broken
# - 0.22-0.23 triggers deprecation warnings by calling its own functions
# - 0.21 is the most stable version
# - <= 0.20 is missing dict methods for gym.spaces.Dict
# - 0.18-0.21 require setuptools<=65.5.0

# Extensions
class BuildExt(build_ext):
    def run(self):
        # Propagate any build_ext options (e.g., --inplace, --force) to subcommands
        build_ext_opts = self.distribution.command_options.get('build_ext', {})
        if build_ext_opts:
            # Copy flags so build_torch and build_c respect inplace/force
            self.distribution.command_options['build_torch'] = build_ext_opts.copy()
            self.distribution.command_options['build_c'] = build_ext_opts.copy()

        self.run_command('build_torch')
        self.run_command('build_c')

class CBuildExt(build_ext):
    def run(self, *args, **kwargs):
        self.extensions = [e for e in self.extensions if not (e.name == "pufferlib._C" or e.name == "pufferlib.native")]
        native_so = None
        if not NO_TORCH:
            self.run_command('build_torch')
            native_so = _find_built_pufferlib_native(required=True)
            print(f"Found pufferlib.native extension at: {native_so}")
            for ext in (self.distribution.ext_modules or []):
                print(f"Checking extension: {ext.name}")
                if getattr(ext, "name", "").startswith("pufferlib.ocean."):
                    ext.extra_objects = list(getattr(ext, "extra_objects", []) or [])
                    if native_so not in ext.extra_objects:
                        print(f"-- Adding native library {native_so} to extension {ext.name}")
                        ext.extra_objects.append(native_so)
        super().run(*args, **kwargs)

class TorchBuildExt(cpp_extension.BuildExtension):
    def run(self):
        self.extensions = [e for e in self.extensions if (e.name == "pufferlib._C" or e.name == "pufferlib.native")]
        super().run()

INCLUDE = [f'{BOX2D_NAME}/include', f'{BOX2D_NAME}/src' ]
RAYLIB_A = f'{RAYLIB_NAME}/lib/libraylib.a'
torch_lib_dirs = torch.utils.cpp_extension.library_paths()
torch_rpaths = [f'-Wl,-rpath,{path}' for path in torch_lib_dirs]

# TODO: CMake or other tools will do this way better and cross-platform too :(
def _find_built_pufferlib_native(required: bool = True):
    ext_suffix = ".so"

    inplace = os.path.join("pufferlib", "native" + ext_suffix)
    if os.path.isfile(inplace):
        return inplace

    cwd = os.getcwd()
    candidates = glob.glob(os.path.join(cwd, "build", "**", "pufferlib", "native*.so"), recursive=True)
    candidates += glob.glob(os.path.join(cwd, "pufferlib", "native*.so"), recursive=True)
    candidates = [p for p in candidates if os.path.isfile(p)]
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    if required:
        raise ValueError(f"Could not find built pufferlib.native extension under {cwd}.")
    return None

# Ensure Ocean env extensions can find pufferlib/native*.so at runtime.
# binding*.so lives under pufferlib/ocean/<env>/; native*.so lives under pufferlib/.
# Relative path from $ORIGIN to pufferlib/ is ../..
origin_rpath = []
if platform.system() == "Linux":
    origin_rpath = ['-Wl,-rpath,$ORIGIN/../..']

extension_kwargs = dict(
    include_dirs=INCLUDE,
    library_dirs=torch_lib_dirs,
    libraries=['torch', 'torch_cpu', 'c10'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args + torch_rpaths + origin_rpath,
    extra_objects=[RAYLIB_A],  # NOTE: native*.so will be injected after build_torch
)

native_lib = _find_built_pufferlib_native(required=False)
if native_lib:
    print(f"Adding native library {native_lib} to C/C++ extensions")
    extension_kwargs['extra_objects'].append(native_lib)

# Find C extensions
c_extensions = []
c_extension_paths = []
if not NO_OCEAN:
    c_extension_paths = glob.glob('pufferlib/ocean/**/binding.c', recursive=True)
    c_extensions = [
        CppExtension(
            path.rstrip('.c').rstrip('.cpp').replace('/', '.'),
            sources=[path, 'pufferlib/ocean/puffer_native_eval.cpp'],
            language='c++',
            **extension_kwargs,
        )
        for path in c_extension_paths if '/breakout' in path or '/grid' in path or '/go' in path or '/g2048' in path or '/pacman' in path or '/blastar' in path or '/pong' in path
    ]
    c_extension_paths = [os.path.join(*path.split('/')[:-1]) for path in c_extension_paths]

    # If you have per-env extra_objects (e.g., box2d), keep doing that here:
    for c_ext in c_extensions:
        if "impulse_wars" in c_ext.name:
            c_ext.extra_objects.append(f'{BOX2D_NAME}/libbox2d.a')
            # TODO: Figure out why this is necessary for some users
            impulse_include = 'pufferlib/ocean/impulse_wars/include'
            if impulse_include not in c_ext.include_dirs:
                c_ext.include_dirs.append(impulse_include)

        if 'matsci' in c_ext.name:
            c_ext.include_dirs.append('/usr/local/include')
            c_ext.extra_link_args.extend(['-L/usr/local/lib', '-llammps'])

# Define cmdclass outside of setup to add dynamic commands
cmdclass = {
    "build_ext": BuildExt,
    "build_torch": TorchBuildExt,
    "build_c": CBuildExt,
}

if not NO_OCEAN:
    def create_env_build_class(full_name):
        class EnvBuildExt(build_ext):
            def run(self):
                self.extensions = [e for e in self.extensions if e.name == full_name]
                super().run()
        return EnvBuildExt

    # Add a build_<env> command for each env
    for c_ext in c_extensions:
        env_name = c_ext.name.split('.')[-2]
        cmdclass[f"build_{env_name}"] = create_env_build_class(c_ext.name)


# Check if CUDA compiler is available. You need cuda dev, not just runtime.
torch_extensions = []
if not NO_TRAIN:
    torch_sources = [
        "pufferlib/extensions/pufferlib.cpp",
    ]
    torch_extensions = []
    if BUILD_CUDA_EXT:
        extension = CUDAExtension
        torch_sources += [
            "pufferlib/extensions/cuda/pufferlib.cu",
            "pufferlib/puffer_cuda_kernels.cu"
        ]
        torch_extensions += [
           extension(
                "pufferlib.native",
                [
                    "pufferlib/puffer_cuda_kernels.cu",
                    "pufferlib/ocean/puffer_cuda.cpp",
                ],
                extra_compile_args = {
                    "cxx": cxx_args,
                    "nvcc": nvcc_args,
                }
            ),
        ]
    else:
        extension = CppExtension
    if NO_PUFFERLIB:
        print("Skipping building pufferlib._C extension as NO_PUFFERLIB is set.")
    else:    
        torch_extensions += [
           extension(
                "pufferlib._C",
                torch_sources,
                extra_compile_args = {
                    "cxx": cxx_args,
                    "nvcc": nvcc_args,
                }
            ),
        ]

# Prevent Conda from injecting garbage compile flags
from distutils.sysconfig import get_config_vars
cfg_vars = get_config_vars()
for key in ('CC', 'CXX', 'LDSHARED'):
    if cfg_vars[key]:
        cfg_vars[key] = cfg_vars[key].replace('-B /root/anaconda3/compiler_compat', '')
        cfg_vars[key] = cfg_vars[key].replace('-pthread', '')
        cfg_vars[key] = cfg_vars[key].replace('-fno-strict-overflow', '')

for key, value in cfg_vars.items():
    if value and '-fno-strict-overflow' in str(value):
        cfg_vars[key] = value.replace('-fno-strict-overflow', '')

install_requires = [
    'setuptools',
    'numpy<2.0',
    'shimmy[gym-v21]',
    'gym==0.23',
    'gymnasium>=0.29.1',
    'pettingzoo>=1.24.1',
]

if not NO_TRAIN:
    install_requires += [
        'torch',
        'psutil',
        'nvidia-ml-py',
        'rich',
        'rich_argparse',
        'imageio',
        'gpytorch',
        'scikit-learn',
        'heavyball>=2.2.0', # contains relevant fixes compared to 1.7.2 and 2.1.1
        'neptune',
        'wandb',
    ]
setup(
    version="3.0.0",
    packages=find_namespace_packages() + find_packages() + c_extension_paths + ['pufferlib/extensions'],
    package_data={
        "pufferlib": [RAYLIB_NAME + '/lib/libraylib.a']
    },
    include_package_data=True,
    install_requires=install_requires,
    ext_modules = torch_extensions + c_extensions,
    cmdclass=cmdclass,
    include_dirs=[numpy.get_include(), 
                  RAYLIB_NAME + '/include', 
                  'pufferlib/ocean', 
                  'pufferlib/extensions', 
                  pybind11.get_include(), 
                  ] + CUDA_INCLUDE,
)
