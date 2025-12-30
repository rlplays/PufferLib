printf '\e]11;#1e1e1e\a'
git clone https://github.com/rlplays/PufferLib.git 
python -m venv puffenv
source puffenv/bin/activate

cd PufferLib
mkdir -p experiments
pwd
git checkout puffer-mt-evallibtorch

sudo apt-get update

nvcc --version
nvidia-smi 
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Important as we are using a venv, otherwise we face cuda 12.8 vs cuda 13.0 nightmare 
# pip3 install -e . --no-build-isolation
pip install uv ninja
# May be try uv pip install -e . -v --no-deps 
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

# Then install your repo editable, compiling extensions against that torch
uv pip install torch setuptools wheel Cython numpy torch pybind11 gymnasium psutil gpytorch rich rich_argparse heavyball torchvision torchaudio nvidia-ml-py 'gym==0.23'
uv pip install -e . --no-build-isolation -v
# IF torch fails with 12.8 vs 13.0  
# uv pip install --force-reinstall --no-build-isolation -v heavyball
mkdir -p experiments
MAX_JOBS=16  bash scripts/build_envs.sh breakout
MAX_JOBS=16  bash scripts/build_envs.sh go