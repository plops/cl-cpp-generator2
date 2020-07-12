git clone https://github.com/opencollab/arpack-ng
cd arpack-ng
mkdir b
cd b
cmake -G Ninja  \
      -D CMAKE_CXX_FLAGS="-Ofast -march=native -mtune=native" \
      -D CMAKE_C_FLAGS="-Ofast -march=native -mtune=native" \
      -D CMAKE_Fortran_FLAGS="-Ofast -march=native -mtune=native" \
      -D ICB=ON -D EXAMPLES=ON ..
# note: -Ofast enables optimizations that are not valid for all standard compliant programs
# -D BUILD_SHARED_LIBS=ON
ninja 
