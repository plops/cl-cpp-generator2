export NUM="00"
export ODIR=/home/martin/stage/cl-cpp-generator2/example/72_emsdk/$NUM"source/"

cd $ODIR

mkdir b

source "/home/martin/src/emsdk/emsdk_env.sh"
emcmake cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug
