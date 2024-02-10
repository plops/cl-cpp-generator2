- try out daxa https://tutorial.learndaxa.com/ an opinionated GPU API
  abstraction over Vulkan

- it requires VulkanMemoryAllocator, a library keeps track of
  allocated memory blocks, used and unused ranges inside them, finds
  best matching unused ranges for new allocations.  helps to choose
  correct and optimal memory type based on intended usage of the
  memory.

- install dependencies VMA, Daxa and libfmt
```
cd ~/src
git clone https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator # 32MB
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/vulkan
ninja install # no build required

cd ~/src
git clone https://github.com/glfw/glfw
mkdir -p glfw/build
cd glfw/build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/vulkan
ninja
ninja install


git clone https://github.com/Ipotrick/Daxa # 15MB
cd Daxa
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=~/vulkan \
  -DDAXA_ENABLE_UTILS_TASK_GRAPH=ON \
  -DDAXA_ENABLE_UTILS_PIPELINE_MANAGER_GLSLANG=ON 
  
time ninja # 8sec
ninja install


cd ~/src
git clone https://github.com/fmtlib/fmt # 15MB
mkdir -p fmt/build
cd fmt/build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/vulkan
time ninja
ninja install
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
    vulkan-config.cmake
```


- vscode extensions:
```
C/C++ Extension Pack (ms-vscode.cpptools-extension-pack)
GLSL Lint (dtoplak.vscode-glsllint)
```
