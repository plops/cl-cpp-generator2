export NUM="04"
export ODIR=/home/martin/stage/cl-cpp-generator2/example/72_emsdk/$NUM"source/"

mkdir -p $ODIR/b_linux

cd $ODIR/b_linux

cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    ..
ninja -v


