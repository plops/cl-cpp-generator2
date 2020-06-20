#ifndef PROTO2_H
#define PROTO2_H
int main ();  
template<typename... ARGS> explicit Code (ARGS&& ...args);  
static Code FromFile (const std::string& name);  
const auto& code ();  
template<typename... ARGS> explicit Header (const std::string& name, ARGS&& ...args);  
const auto& name ();  
 Program (const std::string& name, const Code& code);  
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