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
         CudaDeviceProperties (const cudaDeviceProp& props)  ;  
        public:
         CudaDeviceProperties (int device)  ;  
        static CudaDeviceProperties FromExistingProperties (const cudaDeviceProp& props)  ;  
};
class CudaDevice  {
            int _device ;
    CudaDeviceProperties _props ;
        public:
        explict  CudaDevice (int device)  ;  
        inline CUdevice handle () const ;  
};
#endif