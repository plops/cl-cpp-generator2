CppCon 2017： Pablo Halpern “Allocators： The Good Parts” [v3dz-AKOVL8]

32:08 Test resource 


C++17's std::pmr comes with a cost

synchronized_pool_resource .. for data frequently accessed at the same time
unsynchronized_pool_resoruce .. no thread sync cost
monotonic_buffer_resource .. high speed, high space
