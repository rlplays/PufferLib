echo "Training using: "
cat ./pufferlib/config/ocean/grid_interact.ini
python -m pufferlib.pufferl train puffer_grid_interact --train.device cuda   
if [ $? -ne 0 ]; then
  echo "Unable to train"
  exit 1
fi

echo "Done training..."


echo "Eval'ing trained model..."

python -m pufferlib.pufferl eval puffer_grid_interact --train.device cuda --load-model-path latest    




echo "Exporting weights..."
sh pufferlib/ocean/grid_interact/export_weights.sh
