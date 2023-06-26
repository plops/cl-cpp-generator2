https://github.com/bkryza/clang-uml/

```
cd ~/src
git clone https://github.com/bkryza/clang-uml
cd clang-uml
sudo pacman -S llvm yaml-cpp
#make release

mkdir b
cd b
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..

```
