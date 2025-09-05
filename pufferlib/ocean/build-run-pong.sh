#!/bin/bash
echo "NOTE: Using raylib from ../../../raylib/ - make sure it is present there first."
cmake -S ./pong/ -B build  -DFETCHCONTENT_SOURCE_DIR_RAYLIB=../../../raylib/ $CMAKE_ARGS 
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
