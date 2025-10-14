# Swimming in the Ocean

Ocean lets you build C environments with some binding Python glue to train/eval using PufferLib.
Once you train the models, the exported weights can then be used by the C environment with only the `puffernet.h` dependency.

As always, follow the docs in [the official docs](https://puffer.ai/docs.html). 

This doc is aimed more towards folks unfamiliar with Python/research environments and for those using raw source instead of prebuilt images/Docker.


### Quick notes

- Use a physical Linux machine if possible (HyperV/VM might be hard to configure - but you can get it to work). WSL on Windows works fine.
- Do not use `conda` as it slows down. Use venv:

```
cd PufferLib
python -m venv puffenv
source puffenv/bin/activate

# To deactivate, run `deactivate` to pop out of venv
```


## Try a built-in Ocean environment first

### Compile/run raw demo  - `Target`

On Ubuntu/WSL (tested on Ubuntu 22.04 but 24 should be fine too):

```sh
bash scripts/build_ocean.sh target
./target
```

...should show a [raylib](https://raylib.com) window with puffer fish eating the stars.

## Train and eval the agent using PufferLib

Next, build pufferlib from source and train/eval the

```sh
# First install any deps as needed (especially if you get errors like "#include <Python.h><-- not found")
sudo apt-get update
sudo apt-get install python3-dev build-essential  nvidia-cuda-toolkit 

pip install -e .
# Clear the pip cache if needed `pip cache dir` and remove that dir.

# Then use Ocean envs
python setup.py build_ext --inplace --force

# Should take about 2 mins on our machine.
python -m pufferlib.pufferl train puffer_target --train.device cuda

python -m pufferlib.pufferl eval puffer_target --train.device cuda --load-model-path latest

```

If CUDA fails, try these commands:

```
nvcc --version
nvidia-smi

python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Tip: If `--train.device cuda` doesn't work, try `--train.device cpu`. It's much slower but it's a good start. However, it's highly recommended to using a graphics card to train.

Notes:
- [target.py](../pufferlib/ocean/target/target.py) is used by the train/eval with [binding.c](../pufferlib/ocean/target/binding.c) that interfaces with the actual environment in [target.h](../pufferlib/ocean/target/target.h).
- [target.c](../pufferlib/ocean/target/target.c) is a pure demo-only code that is NOT used by the train/eval steps. 
  - This is the standalone code you will use to load the model via `puffernet.h` (or if you are using an env like Squared, no deps at all) but not used during train/eval steps. Likely the part that you can 'ship' publicly with the trained [model](../resources/target/target_weights.bin) loaded from your [resources/](../resources/target/) directory.
- [target.ini](../config/ocean/target.ini) is a config used by the train/eval steps to run the environment/agent etc.


## Try the raw demo using the trained model

Now you are ready to use the model trained / eval'ed earlier:

```sh
# Export weights
python -m pufferlib.pufferl export puffer_target --load-model-path latest
cp -L 

bash scripts/build_ocean.sh target
./target
```

To test out the full train/eval/run steps, try making a simple change in `target.ini` to set `num_agents` and `num_goals` `= 1` and re-train/eval/run the demo to see that it works. (Make sure to re-train otherwise the `torch` shape won't match the weights)


## Build, Train, Eval and Run your own Ocean environment

Here is a simplified checklist to copy the `target` sample env to a `newenv` (name accordingly).
(Tip: You can copy paste the entire checklist, s/newenv/your_actual_env/ and run the commands too).

- [ ] Copy [target](../pufferlib/ocean/target/) to [pufferlib/ocean/newenv](../pufferlib/ocean/)
  - [ ] `mkdir -p ./pufferlib/ocean/newenv/ && cp -L ./pufferlib/ocean/target/* ./pufferlib/ocean/newenv/`
- [ ] Rename files from `target.*` to `newenv.*` in the new directory
- [ ] Update references in the code from `target` / `Target` to `newenv` / `NewEnv`
- [ ] Create a new config file `newenv.ini` in the [config](../pufferlib/config/ocean/) directory
  - [ ] Note: Must have `puffer_` in the name if you want this to be an environment in `ocean/`
  - [ ] `cp -L ./pufferlib/config/ocean/target.ini ./pufferlib/config/ocean/newenv.ini`
  - NOTE: [`../config/`](../config/) is a symlink to [`../pufferlib/config/*`](../pufferlib/config/)
- [ ] Update the binding and header files with your environment logic
- [ ] Add to the env list in [environment.py](../pufferlib/ocean/environment.py)
- [ ] Update the environment logic in `newenv.c` and test compilation with `bash scripts/build_ocean.sh newenv`
- [ ] Train with `python -m pufferlib.pufferl train puffer_newenv --train.device cuda`
- [ ] Eval with `python -m pufferlib.pufferl eval puffer_newenv --train.device cuda`
- [ ] Test the program with the loaded weights using
  - [ ] `python -m pufferlib.pufferl export puffer_newenv --load-model-path latest`
  - [ ] `mv puffer_newenv_weights.bin ./resources/newenv/newenv_weights.bin`
  - [ ] `bash scripts/build_ocean.sh newenv`
  - [ ] `./newenv`


## Debug your Ocean environment

- Ensure `c_envs` is filled correctly. Output a single frame in `pufferl.py`'s `train()` during training and ensure it matches the `num_envs` for `o, r, t` nd arrays.

  - If multi-agent, the `c_envs` contains `num_agents*num_envs` envs.
    - From  Puffer perspective, it's simply a single-agent, just happens to train multiple actions/observations.

- Ensure `c_reset` resets only the environment and not the `log` (and that the perf/scores are accumulated correctly before resetting)
  - Also don't allocate anything extra in `c_reset` - it's meant to reset the agent in the _existing_ environment not necessarily "load a new level".

