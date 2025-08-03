sh build.sh
if [ $? -ne 0 ]; then
  exit 1
fi

sh train.sh
if [ $? -ne 0 ]; then
  exit 1
fi

