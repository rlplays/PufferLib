#!/bin/bash

python setup.py build_torch
IFS=',' read -ra envs <<< "$1"
for env in "${envs[@]}"; do

  echo "Building env: $env"
  python setup.py build_"$env" --inplace --force || read -r -p "Build failed for $env. Press Enter to continue..." _
done
