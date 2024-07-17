
```
cd ~/src

git clone --recursive https://github.com/stephenrkell/toolsub
sudo apt install ocaml ocaml-findlib
cd toolsub
make

# Complains about missing CIL, no idea how to proceed

git clone https://github.com/stephenrkell/libcxxfileno
cd libcxxfileno
libtoolize && autoreconf -i && ./configure --prefix=/home/kiel/amd64_cxxfileno && make && make install


git clone https://github.com/stephenrkell/liballocs
cd liballocs
./autogen.sh
sudo apt install libunwind-dev binutils-dev libboost-regex-dev libboost-system-dev libboost-filesystem-dev
ln -s /home/kiel/amd64_cxxfileno/lib/pkgconfig/libc++fileno.pc /home/kiel/amd64_cxxfileno/lib/pkgconfig/libcxxfileno.pc
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/kiel/amd64_cxxfileno/lib/pkgconfig
./configure --prefix=/home/kiel/amd64_liballocs
```
