//
// Created by martin on 9/25/24.
// i used qwen2.5
//
#include <iostream>
#include <vector>
#include <memory_resource>
#include <cstdlib>

// Custom memory resource for Point2D arrays
class Point2DMemoryResource : public std::pmr::memory_resource {
public:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        return std::malloc(bytes); // Using malloc for simplicity
    }

    void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) override {
        std::free(ptr);
    }

    bool is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }
};

int main() {
    // Create a custom memory resource for Point2D arrays
    Point2DMemoryResource point2d_pool;
3
    // Use the custom memory resource with std::pmr::vector
    std::pmr::vector<Point2D, std::pmr::memory_resource*> point2ds{&point2d_pool};

    // Allocate and initialize some Point2D objects
    for (int i = 0; i < 5; ++i) {
        point2ds.push_back({static_cast<float>(i), static_cast<float>(i * 1.1)});
    }

    // Print the contents of the vector
    for (const auto& p : point2ds) {
        std::cout << "Point2D(x=" << p.x << ", y=" << p.y << ")" << std::endl;
    }

    return 0;
}