
python -m pufferlib.pufferl export puffer_grid_interact --load-model-path latest
if [ $? -ne 0 ]; then
  echo "Export failed"
  exit 1
fi

