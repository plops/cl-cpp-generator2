https://github.com/bkryza/clang-uml/

```
cd ~/src
git clone https://github.com/bkryza/clang-uml
cd clang-uml
sudo pacman -S llvm yaml-cpp


cmake -G Ninja \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=/home/martin/clang-uml \
-DBUILD_TESTS=OFF \
-DCMAKE_CXX_COMPILER=clang++ \
..


```
