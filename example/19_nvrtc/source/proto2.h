#ifndef PROTO2_H
#define PROTO2_H
int main ();  
template<typename... ARGS> explicit Code (ARGS&& ...args);  
const auto& code ();  
explicit CudaDeviceProperties (const cudaDeviceProp& props);  
 CudaDeviceProperties (int device);  
static CudaDeviceProperties FromExistingProperties (const cudaDeviceProp& props);  
static CudaDeviceProperties ByIntegratedType (bool integrated);  
const auto& getRawStruct ();  
auto major ();  
auto minor ();  
bool integrated ();  
const char* name ();  
explicit CudaDevice (int device);  
inline CUdevice handle ();  
static CudaDevice FindByProperties (const CudaDeviceProperties& props);  
static int NumberOfDevices ();  
void setAsCurrent ();  
const auto & properties ();  
const char* name ();  
static CudaDevice FindByName (std::string name);  
static std::vector<CudaDevice> EnumerateDevices ();  
static CudaDevice CurrentDevice ();  
 CudaContext (const CudaDevice& device);  
 ~CudaContext ();  
#endif