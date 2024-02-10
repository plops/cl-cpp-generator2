mkdir b
cd b
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="/home/martin/vulkan/share/daxa;/home/martin/vulkan/share/cmake/VulkanMemoryAllocator;/home/martin/vulkan/lib64/cmake/fmt;/home/martin/vulkan/lib64/cmake/glfw3"
