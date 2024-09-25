//
// Created by martin on 9/25/24.
// i used qwen2.5
// and klaus iglberger: c++ software design
//
#include <iostream>
#include <vector>
#include <memory_resource>
#include <cstdlib>
#include <array>

// Custom memory resource for Point2D arrays
class Point2DMemoryResource : public std::pmr::memory_resource {
public:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        return std::malloc(bytes); // Using malloc for simplicity
    }

    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override {
        std::free(ptr);
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }
};
struct Point2D { // size of this structure is 8 bytes
    float x;
    float y;

    Point2D(float x = 0.0f, float y = 0.0f) : x(x), y(y) {}
};
int main() {
    std::array<std::byte, 24> raw{}; // 24 bytes should be enough but we need 56 bytes. probably std::vector allocates more
    // the pool is limited to <x> bytes, trying to allocate more will throw
    std::pmr::monotonic_buffer_resource
      buffer{raw.data(), raw.size(), std::pmr::null_memory_resource()};
    // std::vector<Point2D, std::pmr::polymorphic_allocator<Point2D> > points{&buffer};
    std::pmr::vector<Point2D> points{&buffer};
    points.emplace_back(.1, .2);
    points.emplace_back(.3, .4);
    points.emplace_back(.5, .6);



    // Print the contents of the vector
//    for (const auto& p : point2ds) {
//        std::cout << "Point2D(x=" << p.x << ", y=" << p.y << ")" << std::endl;
//    }

    return 0;
}