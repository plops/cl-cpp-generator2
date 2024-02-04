mkdir bclang
cd bclang
export CC=clang
export CXX=clang++
cmake -G Ninja \
 -DCMAKE_BUILD_TYPE=Release \
 -DENABLE_RYZEN_TESTS=OFF \
 -DBUILD_EXAMPLE=OFF \
 -DBUILD_GMOCK=OFF \
 -DBUILD_TESTS=OFF \
 -DGLFW_BUILD_WAYLAND=OFF \
 -DDEP_DIR=/home/martin/src \
 ..
