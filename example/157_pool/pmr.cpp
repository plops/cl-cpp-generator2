//
// Created by martin on 9/25/24.
// i used qwen2.5
// and klaus iglberger: c++ software design
//
#include <iostream>
#include <vector>
#include <memory_resource>
#include <array>

struct Point2D {
    double x;
    double y;
    Point2D(double x = 0.0f, double y = 0.0f) : x(x), y(y) {}
};
int main() {
    const int n = 3;
    std::array<std::byte, n*sizeof(Point2D)> raw{};  // the pool is limited to <x> bytes, trying to allocate more will throw
    std::pmr::monotonic_buffer_resource
      buffer{raw.data(), raw.size(), std::pmr::null_memory_resource()};
    // specify the vector size explicitly to prevent running out of memory from the pool
    std::pmr::vector<Point2D> points{n,&buffer};
    points[0] = {.1, .2};
    points[1] = {.3, .4};
    points[2] = {.5, .6};


    return 0;
}