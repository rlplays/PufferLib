#!/bin/bash
# e.g. usage: bash profile_env.sh puffer_go,puffer_breakout,puffer_squared --profile.name multiproc --profile.train 0

IFS=',' read -ra envs <<< "$1"
mkdir -p experiments/
for env in "${envs[@]}"; do
   echo "Using env: $env"
   echo "python -m pufferlib.pufferl profile $env --train.device cuda ${@:2}"
   # To debug uncomment this instead:
   # python -m debugpy --wait-for-client --listen 0.0.0.0:5678 -m pufferlib.pufferl profile $env --train.device cuda ${@:2}
   python -m pufferlib.pufferl profile "$env" --train.device cuda "${@:2}"
done