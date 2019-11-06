// https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf magma by example
// g++ -O3 -fopenmp -march=native -std=c++11 -DHAVE_CUBLAS -I/opt/cuda/include -I/usr/local/magma/include -c -o 05magma_isamax.o 05magma_isamax.cpp; g++ -O3 -fopenmp -march=native -L/opt/cuda/lib64 -L/usr/local/magma/lib -lmagma -lopenblas -lcublas -lcudart -o 05magma_isamax 05magma_isamax.o; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/magma/lib/
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <magma_v2.h>
int main (int argc, char** argv){
        magma_init();
            auto queue  = (magma_queue_t) NULL;
    auto dev  = magma_int_t(0);
    auto m  = magma_int_t(1024);
    float* a ;
    magma_queue_create(dev, &queue);
    cudaMallocManaged(&a, ((m)*(sizeof(float))));
    for (int j = 0;j<m;(j)+=(1)) {
                        a[j]=sinf(float(j));
}
        auto i  = magma_isamax(m, a, 1, queue);
    cudaDeviceSynchronize();
        magma_free(a);
    magma_queue_destroy(queue);
    magma_finalize();
    return 0;
}