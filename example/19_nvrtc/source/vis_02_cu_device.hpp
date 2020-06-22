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
        static CudaDeviceProperties FromExistingProperties (const cudaDeviceProp& props);  
        static CudaDeviceProperties ByIntegratedType (bool integrated);  
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
        static CudaDevice FindByProperties (const CudaDeviceProperties& props);  
        static int NumberOfDevices ();  
        void setAsCurrent ();  
        const auto & properties ();  
        const char* name ();  
        static CudaDevice FindByName (std::string name);  
        static std::vector<CudaDevice> EnumerateDevices ();  
        static CudaDevice CurrentDevice ();  
};
class CudaContext  {
            CUcontext _ctx ;
        public:
         CudaContext (const CudaDevice& device);  
         ~CudaContext ();  
};
#endif