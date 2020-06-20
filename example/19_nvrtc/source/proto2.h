#ifndef PROTO2_H
#define PROTO2_H
int main ();  
explicit CudaDeviceProperties (const cudaDeviceProp& props);  
 CudaDeviceProperties (int device);  
static CudaDeviceProperties FromExistingProperties (const cudaDeviceProp& props);  
static CudaDeviceProperties ByIntegratedType (bool integrated);  
const auto& getRawStruct ();  
auto major ();  
auto minor ();  
bool integrated ();  
const char* name ();  
#endif