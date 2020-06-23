#ifndef VIS_02_CU_DEVICE_H
#define VIS_02_CU_DEVICE_H
#include "utils.h"
;
#include "globals.h"
;
#include <cuda_runtime.h>
#include <cuda.h>
;
#include <algorithm>
#include <vector>
;
#include "vis_02_cu_device.hpp"
;
class CudaDeviceProperties  {
            cudaDeviceProp _props ;
         CudaDeviceProperties (const cudaDeviceProp& props);  
        public:
         CudaDeviceProperties (int device);  
        CudaDeviceProperties FromExistingProperties (const cudaDeviceProp& props);  
        CudaDeviceProperties ByIntegratedType (bool integrated);  
        const auto& getRawStruct ();  
        auto major ();  
        auto minor ();  
        bool integrated ();  
        const char* name ();  
};
class CudaDevice  {
            int _device ;
    CudaDeviceProperties _props ;
        public:
         CudaDevice (int device);  
        inline CUdevice handle ();  
        CudaDevice FindByProperties (const CudaDeviceProperties& props);  
        int NumberOfDevices ();  
        void setAsCurrent ();  
        const auto & properties ();  
        const char* name ();  
        CudaDevice FindByName (std::string name);  
        std--vector<CudaDevice> EnumerateDevices ();  
        CudaDevice CurrentDevice ();  
};
class CudaContext  {
            CUcontext _ctx ;
        public:
         CudaContext (const CudaDevice& device);  
         ~CudaContext ();  
};
#endif