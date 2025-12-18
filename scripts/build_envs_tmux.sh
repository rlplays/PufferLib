#!/bin/bash

IFS=',' read -ra envs <<< "$1"
for env in "${envs[@]}"; do
  echo "Building env: $env in tmux pane"
  tmux split-window -h "python setup.py build_$env --inplace --force || read -r -p 'Build failed for $env. Press Enter to continue...'"
done
