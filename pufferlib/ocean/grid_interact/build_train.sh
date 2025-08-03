sh ./pufferlib/ocean/grid_interact/build.sh
if [ $? -ne 0 ]; then
  exit 1
fi

sh ./pufferlib/ocean/grid_interact/train.sh
if [ $? -ne 0 ]; then
  exit 1
fi

