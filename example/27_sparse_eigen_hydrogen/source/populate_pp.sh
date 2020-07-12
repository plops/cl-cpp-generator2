git clone https://github.com/m-reuter/arpackpp
cd arpackpp
mkdir b
cd b
cmake -G Ninja  \
      -D CMAKE_CXX_FLAGS="-Ofast -march=native -mtune=native" \
      -D CMAKE_C_FLAGS="-Ofast -march=native -mtune=native" \
      -D CMAKE_Fortran_FLAGS="-Ofast -march=native -mtune=native" \
      -D CMAKE_BUILD_TYPE="Release" \
      -D ARPACK_LIB=../../arpack-ng/b/libarpack.a \
      ..
ninja 
