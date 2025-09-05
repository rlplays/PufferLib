cmake -S ./pong/ -B build $CMAKE_ARGS
if [ $? -ne 0 ]; then
  echo "Build failed"
  exit 1
fi
cmake --build ./build -j
if [ $? -ne 0 ]; then
  echo "Build failed"
  exit 1
fi
./pong $*
