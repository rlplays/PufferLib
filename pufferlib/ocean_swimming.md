# Swimming in the Ocean

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

On WSL (I am using Ubuntu 22.04 but 24 should be fine too)

```
bash scripts/build_ocean.sh target
./target
```

## Train the agent/eval

Next, build pufferlib from source and train/eval the

```
pip install -e .
# Clear the pip cache if needed `pip cache dir` and remove that dir.

# Then use Ocean envs
python setup.py build_ext --inplace --force

# Should take about 2 mins on our machine.
python -m pufferlib.pufferl train puffer_target --train.device cuda

python -m pufferlib.pufferl eval puffer_target --train.device cuda --load-model-path latest

```



## Build an Ocean environment