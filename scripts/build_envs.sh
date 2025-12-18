#!/bin/bash

IFS=',' read -ra envs <<< "$1"
for env in "${envs[@]}"; do
  echo "Building env: $env"
  tmux split-window -h "python setup.py build_$env --inplace --force"
done
