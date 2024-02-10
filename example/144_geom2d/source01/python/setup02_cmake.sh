wget https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.tar.gz
tar xaf v2.11.1.tar.gz
mkdir -p build2
cd build2
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
ninja
cp olcUTIL_Geometry2D_py.* ..
cd ..
python test_python.py
