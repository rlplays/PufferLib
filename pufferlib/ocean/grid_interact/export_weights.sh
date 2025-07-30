
python -m pufferlib.pufferl export puffer_grid_interact --load-model-path latest
mv puffer_grid_interact_weights.bin ./resources/grid_interact/grid_interact_weights.bin
echo "Weights copied to ./resources/grid_interact/grid_interact_weights.bin"