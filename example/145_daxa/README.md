- try out daxa https://tutorial.learndaxa.com/ an opinionated GPU API
  abstraction over Vulkan

- it requires VulkanMemoryAllocator, a library keeps track of
  allocated memory blocks, used and unused ranges inside them, finds
  best matching unused ranges for new allocations.  helps to choose
  correct and optimal memory type based on intended usage of the
  memory.

- install 
```
cd ~/src
git clone https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator # 32MB
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/vulkan
ninja install # no build required
git clone https://github.com/Ipotrick/Daxa # 15MB
cd Daxa
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/vulkan
time ninja # 8sec
ninja install

martin@archlinux ~/src/Daxa/build $ du -hs ~/vulkan/
1.9M    /home/martin/vulkan/

cd ~/src 
wget https://sdk.lunarg.com/sdk/download/1.3.275.0/linux/vulkansdk-linux-x86_64-1.3.275.0.tar.xz # 247M
tar xaf vulkansdk-linux-x86_64-1.3.275.0.tar.xz
cd 1.3.275.0 # 1.6GB (binaries)
mkdir build
cd build
cmake .. \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-DNDEBUG" \
  -DCMAKE_CXX_FLAGS="-DNDEBUG" \
  -DCMAKE_SKIP_RPATH=ON \ 
  -DBUILD_TESTS=OFF \
  -DBUILD_WSI_WAYLAND_SUPPORT=OFF \
  -DBUILD_WSI_XCB_SUPPORT=ON \
  -DBUILD_WSI_XLIB_SUPPORT=ON \
  -DVULKAN_HEADERS_INSTALL_DIR=~/vulkan \
  -DCMAKE_PREFIX_PATH=~/vulkan \
  -DENABLE_WERROR=OFF
```

```
cd ~/stage/cl-cpp-generator2/example/145_daxa/source01/b
 cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
```
- i'm currently stuck with this error:
```
CMake Error at CMakeLists.txt:17 (find_package):
  Could not find a package configuration file provided by "Vulkan" with any
  of the following names:

    VulkanConfig.cmake
    vulkan-config.cmak
```


- vscode extensions:
```
C/C++ Extension Pack (ms-vscode.cpptools-extension-pack)
GLSL Lint (dtoplak.vscode-glsllint)
```
