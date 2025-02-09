# install dependencies

- install vulkan, glfw, spirv, glslang

```bash
cd ~/src
git clone --recursive https://github.com/Ipotrick/Daxa
git clone --recursive https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
```

patch impl_task_graph of daxa:
```c++
#include <atomic>


#define DAXA_TASK_GRAPH_MAX_CONDITIONALS 31
// typedef std::atomic< uint32_t > atomic_uint32_t
// instead of typedef use more modern approach:
using atomic_uint32_t = std::atomic<uint32_t>;

```