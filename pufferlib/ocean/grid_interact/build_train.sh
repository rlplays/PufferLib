echo "Building grid_interact..."
bash scripts/build_ocean.sh grid_interact 
if [ $? -ne 0 ]; then
  echo "Unable to build grid_interact"
  exit 1
fi

echo "Building ext for training"
python setup.py build_ext --inplace --force 
if [ $? -ne 0 ]; then
  echo "Unable to build ext"
  exit 1
fi

echo "Training using: "
cat ./pufferlib/config/ocean/grid_interact.ini
python -m pufferlib.pufferl train puffer_grid_interact --train.device cuda   
if [ $? -ne 0 ]; then
  echo "Unable to train"
  exit 1
fi

echo "Done training..."

sh pufferlib/ocean/grid_interact/export_weights.sh

echo "Eval'ing trained model..."

python -m pufferlib.pufferl eval puffer_grid_interact --train.device cuda --load-model-path latest    



