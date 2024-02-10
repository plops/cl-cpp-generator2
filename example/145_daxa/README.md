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
```
- vscode extensions:
```
C/C++ Extension Pack (ms-vscode.cpptools-extension-pack)
GLSL Lint (dtoplak.vscode-glsllint)
```
