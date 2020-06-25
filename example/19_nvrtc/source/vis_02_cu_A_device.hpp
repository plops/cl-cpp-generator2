#ifndef VIS_02_CU_A_DEVICE_H
#define VIS_02_CU_A_DEVICE_H
#include "utils.h"
;
#include "globals.h"
;
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
;
#include "vis_02_cu_A_device.hpp"
;
class CudaDeviceProperties  {
            cudaDeviceProp _props ;
        explicit  CudaDeviceProperties (const cudaDeviceProp& props)  ;  ;
        public:
         CudaDeviceProperties (int device)  ;  ;
        static CudaDeviceProperties FromExistingProperties (const cudaDeviceProp& props)  ;  ;
        static CudaDeviceProperties ByIntegratedType (bool integrated)  ;  ;
        const auto& getRawStruct () const ;  ;
        int major () const ;  
        int minor () const ;  
        bool integrated () const ;  
        const char* name () const ;  
};
class CudaDevice  {
            int _device ;
    CudaDeviceProperties _props ;
        public:
        explicit  CudaDevice (int device)  ;  ;
        inline CUdevice handle () const   {
                        CUdevice h ;
        if ( !((CUDA_SUCCESS)==(cuDeviceGet(&h, _device))) ) {
                                    throw std::runtime_error("cuDeviceGet(&h, _device)");
};
        return h;
};
        static CudaDevice FindByProperties (const CudaDeviceProperties& props)  ;  ;
        static int NumberOfDevices ()  ;  ;
        void setAsCurrent ()  ;  ;
        const CudaDeviceProperties & properties () const ;  
        const char* name () const ;  
        static CudaDevice FindByName (std::string name)  ;  ;
        static std::vector<CudaDevice> EnumerateDevices ()  ;  ;
        static CudaDevice CurrentDevice ()  ;  ;
};
#endif