export NUM="04"
export ODIR=/home/martin/stage/cl-cpp-generator2/example/72_emsdk/$NUM"source/"

mkdir -p $ODIR/b

cd $ODIR/b

cp ../index.html .
source "/home/martin/src/emsdk/emsdk_env.sh"
emcmake cmake \
	-G Ninja \
	-DCMAKE_BUILD_TYPE=Debug \
	..
# -DCMAKE_TOOLCHAIN_FILE=/home/martin/src/vcpkg/scripts/buildsystems/vcpkg.cmake 
# -DCMAKE_PREFIX_PATH="/home/martin/src/opencv/build_wasm/" 
cmake --build .

