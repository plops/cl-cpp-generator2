export NUM="01"
export ODIR=/home/martin/stage/cl-cpp-generator2/example/72_emsdk/$NUM"source/"

mkdir -p $ODIR/b

cd $ODIR/b

source "/home/martin/src/emsdk/emsdk_env.sh"
emcmake cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug ..
cmake --build .
