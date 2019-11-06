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
    if ( constexpr(std::is_same<decltype(magma_queue_create(dev, &queue)),void>::value)?magma_queue_create(dev, &queue):"void" ) {
                        auto RES612  = magma_queue_create(dev, &queue);
        (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__func__)<<(" magma_queue_create(dev, &queue): => ")<<(RES612)<<(" ")<<(std::endl);
}
    if ( constexpr(std::is_same<decltype(cudaMallocManaged(&a, ((m)*(sizeof(float))))),void>::value)?cudaMallocManaged(&a, ((m)*(sizeof(float)))):"void" ) {
                        auto RES613  = cudaMallocManaged(&a, ((m)*(sizeof(float))));
        (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__func__)<<(" cudaMallocManaged(&a, ((m)*(sizeof(float)))): => ")<<(RES613)<<(" ")<<(std::endl);
}
    for (int j = 0;j<m;(j)+=(1)) {
                        a[j]=sin(static_cast<float>(j));
}
        auto i  = magma_isamax(m, a, 1, queue);
    if ( constexpr(std::is_same<decltype(cudaDeviceSynchronize()),void>::value)?cudaDeviceSynchronize():"void" ) {
                        auto RES614  = cudaDeviceSynchronize();
        (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__func__)<<(" cudaDeviceSynchronize(): => ")<<(RES614)<<(" ")<<(std::endl);
};
        if ( constexpr(std::is_same<decltype(magma_free(a)),void>::value)?magma_free(a):"void" ) {
                        auto RES615  = magma_free(a);
        (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__func__)<<(" magma_free(a): => ")<<(RES615)<<(" ")<<(std::endl);
}
    if ( constexpr(std::is_same<decltype(magma_queue_destroy(queue)),void>::value)?magma_queue_destroy(queue):"void" ) {
                        auto RES616  = magma_queue_destroy(queue);
        (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__func__)<<(" magma_queue_destroy(queue): => ")<<(RES616)<<(" ")<<(std::endl);
}
    if ( constexpr(std::is_same<decltype(magma_finalize()),void>::value)?magma_finalize():"void" ) {
                        auto RES617  = magma_finalize();
        (std::cout)<<(((std::chrono::high_resolution_clock::now().time_since_epoch().count())-(g_start)))<<(" ")<<(__FILE__)<<(":")<<(__LINE__)<<(" ")<<(__func__)<<(" magma_finalize(): => ")<<(RES617)<<(" ")<<(std::endl);
};
    return 0;
}