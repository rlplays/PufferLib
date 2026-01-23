#!/bin/bash

IFS=',' read -ra envs <<< "$1"
for env in "${envs[@]}"; do
  echo "Building: $env"
  python setup.py build_"$env" --inplace --force
  if [ $? -ne 0 ]; then
    echo -e "\033[31m*************************ERRORS***********************************\033[0m"
    echo -e "\033[31merror: Build failed for $env\033[0m"
    echo -e "\033[31m*************************ERRORS***********************************\033[0m"
    exit 1
  fi
  
done
