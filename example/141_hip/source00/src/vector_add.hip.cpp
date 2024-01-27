#ifndef VECTOR_ADD_HIP_H  
#define VECTOR_ADD_HIP_H  
  
#include <hip/hip_runtime.h>  
#include <vector>  
  
constexpr auto N = 512;  
  
// HIP kernel for vector addition  
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements) {  
    auto i = blockDim.x * blockIdx.x + threadIdx.x;  
    if (i < numElements) {  
        C[i] = A[i] + B[i];  
    }  
}  
  
// Function to run the vector addition kernel  
std::vector<float> runVectorAdd(const std::vector<float>& A, const std::vector<float>& B) {  
    assert(A.size() == B.size());  
  
    auto numElements = A.size();  
    std::vector<float> C(numElements);  
  
    float *d_A, *d_B, *d_C;  
    auto size = numElements * sizeof(float);  
  
    // Allocate memory on device  
    hipMalloc(&d_A, size);  
    hipMalloc(&d_B, size);  
    hipMalloc(&d_C, size);  
  
    // Copy data from host to device  
    hipMemcpy(d_A, A.data(), size, hipMemcpyHostToDevice);  
    hipMemcpy(d_B, B.data(), size, hipMemcpyHostToDevice);  
  
    // Launch the kernel  
    auto threadsPerBlock = 256;  
    auto blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;  
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, numElements);  
  
    // Copy result back to host  
    hipMemcpy(C.data(), d_C, size, hipMemcpyDeviceToHost);  
  
    // Free device memory  
    hipFree(d_A);  
    hipFree(d_B);  
    hipFree(d_C);  
  
    return C;  
}  
  
#endif // VECTOR_ADD_HIP_H  
