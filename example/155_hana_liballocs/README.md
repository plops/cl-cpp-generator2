

Try to install dependencies on ubuntu 24x
```
cd ~/src

git clone --recursive https://github.com/stephenrkell/toolsub
sudo apt install ocaml ocaml-findlib
cd toolsub
make

# Complains about missing CIL, no idea how to proceed


sudo apt install libunwind-dev binutils-dev libboost-all-dev
# libboost-regex-dev libboost-system-dev libboost-filesystem-dev libboost-iostreams-dev

git clone --recursive https://github.com/stephenrkell/libdwarfpp
cd libdwarfpp
make -C contrib -j 32 && . contrib/env.sh 
./autogen.sh
./configure --prefix=/home/kiel/amd64_cxxfileno

git clone https://github.com/stephenrkell/libcxxfileno
cd libcxxfileno
libtoolize && autoreconf -i && ./configure --prefix=/home/kiel/amd64_cxxfileno && make -j32 && make install


git clone https://github.com/stephenrkell/libsrk31cxx
cd libsrk31cxx
libtoolize && autoreconf -i && ./configure --prefix=/home/kiel/amd64_cxxfileno && make -j32 && make install



git clone https://github.com/stephenrkell/liballocs
cd liballocs
./autogen.sh

# why do the .pc files contain c++ but configure searches for cxx?
ln -s /home/kiel/amd64_cxxfileno/lib/pkgconfig/libc++fileno.pc /home/kiel/amd64_cxxfileno/lib/pkgconfig/libcxxfileno.pc
ln -s /home/kiel/amd64_cxxfileno/lib/pkgconfig/libsrk31c++.pc /home/kiel/amd64_cxxfileno/lib/pkgconfig/libsrk31cxx.pc
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/kiel/amd64_cxxfileno/lib/pkgconfig
./configure --prefix=/home/kiel/amd64_liballocs
```


found better description, why the 32 bit libraries?
```
sudo apt-get install libelf-dev libdw-dev binutils-dev \
    autoconf automake libtool pkg-config autoconf-archive \
    g++ ocaml ocamlbuild ocaml-findlib \
    default-jre-headless python3 python-is-python3 \
    make git gawk gdb wget \
    libunwind-dev libc6-dev-i386 zlib1g-dev libc6-dbg \
    libboost-{iostreams,regex,serialization,filesystem}-dev && \
git clone --recursive https://github.com/stephenrkell/liballocs.git && \
cd liballocs &&
make -C contrib -j33 && \
./autogen.sh && \
. contrib/env.sh && \
./configure --prefix=/home/kiel/amd64_liballocs && \
make -j32

```
