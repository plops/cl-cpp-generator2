export NUM="04"
export ODIR=/home/martin/stage/cl-cpp-generator2/example/72_emsdk/$NUM"source/"

mkdir -p $ODIR/b_linux

cd $ODIR/b_linux

cp ../index.html .
source "/home/martin/src/emsdk/emsdk_env.sh"
cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    ..
ninja -v


