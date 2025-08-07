echo "Building grid_interact..."
# bash scripts/build_ocean.sh grid_interact fast
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

