#ifndef PROTO2_H
#define PROTO2_H
int main ();  
explicit CudaDeviceProperties (const cudaDeviceProp& props);  
 CudaDevicProperties (int device);  
static FromExistingProperties (const cudaDeviceProp& props);  
static ByIntegratedType (bool integrated);  
const auto& getRawStruct ();  
auto major ();  
auto minor ();  
bool integrated ();  
const char* name ();  
#endif