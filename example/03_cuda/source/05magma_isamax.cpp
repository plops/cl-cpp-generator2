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
            auto RES648  = "void";
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" magma_queue_create(dev, &queue): => ")<<(RES648)<<(" ")<<(std::endl);
            auto RES649  = cudaMallocManaged(&a, ((m)*(sizeof(float))));
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" cudaMallocManaged(&a, ((m)*(sizeof(float)))): => ")<<(RES649)<<(" ")<<(std::endl);
    for (int j = 0;j<m;(j)+=(1)) {
                        a[j]=sin(static_cast<float>(j));
}
        auto i  = magma_isamax(m, a, 1, queue);
            auto RES650  = cudaDeviceSynchronize();
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" cudaDeviceSynchronize(): => ")<<(RES650)<<(" ")<<(std::endl);
                auto RES651  = magma_free(a);
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" magma_free(a): => ")<<(RES651)<<(" ")<<(std::endl);
            auto RES652  = "void";
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" magma_queue_destroy(queue): => ")<<(RES652)<<(" ")<<(std::endl);
            auto RES653  = magma_finalize();
    (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__PRETTY_FUNCTION__)<<(" magma_finalize(): => ")<<(RES653)<<(" ")<<(std::endl);
    return 0;
}