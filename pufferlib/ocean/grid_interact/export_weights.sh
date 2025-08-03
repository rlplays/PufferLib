
python -m pufferlib.pufferl export puffer_grid_interact --load-model-path latest
if [ $? -ne 0 ]; then
  echo "Export failed"
  exit 1
fi

mv puffer_grid_interact_weights.bin ./resources/grid_interact/grid_interact_weights.bin
if [ $? -ne 0 ]; then
  echo "Unable to move weights file"
  exit 1
fi
echo "Weights copied to ./resources/grid_interact/grid_interact_weights.bin"
