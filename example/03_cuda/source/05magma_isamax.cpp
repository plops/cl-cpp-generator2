// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf magma by example
// g++ -O3 -fopenmp -ggdb -march=native -std=c++11 -DHAVE_CUBLAS -I/opt/cuda/include -I/usr/local/magma/include -c -o 05magma_isamax.o 05magma_isamax.cpp; g++ -O3 -fopenmp -march=native -ggdb -L/opt/cuda/lib64 -L/usr/local/magma/lib -lm -lmagma -lopenblas -lcublas -lcudart -o 05magma_isamax 05magma_isamax.o; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/magma/lib/
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <magma_v2.h>
#include <iostream>
#include <chrono>
int main (int argc, char** argv){
            auto g_start  = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        magma_init();
            auto queue  = static_cast<magma_queue_t>(NULL);
    auto dev  = static_cast<magma_int_t>(0);
    auto m  = static_cast<magma_int_t>(1024);
    float* a ;
            auto RES660  = "void";
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" magma_queue_create(dev, &queue): => ")<<(RES660)<<(" ")<<(" dev=")<<(dev)<<(std::endl);
            auto RES661  = cudaMallocManaged(&a, ((m)*(sizeof(float))));
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" cudaMallocManaged(&a, ((m)*(sizeof(float)))): => ")<<(RES661)<<(" ")<<(" m=")<<(m)<<(" a=")<<(a)<<(std::endl);
    for (int j = 0;j<m;(j)+=(1)) {
                        a[j]=sin(static_cast<float>(j));
}
        auto i  = magma_isamax(m, a, 1, queue);
            auto RES662  = cudaDeviceSynchronize();
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" cudaDeviceSynchronize(): => ")<<(RES662)<<(" ")<<(std::endl);
                auto RES663  = magma_free(a);
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" magma_free(a): => ")<<(RES663)<<(" ")<<(std::endl);
            auto RES664  = "void";
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" magma_queue_destroy(queue): => ")<<(RES664)<<(" ")<<(std::endl);
            auto RES665  = magma_finalize();
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" magma_finalize(): => ")<<(RES665)<<(" ")<<(std::endl);
    return 0;
}