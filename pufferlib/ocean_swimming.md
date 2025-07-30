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




## Build an Ocean environment