CXX=/media/sdb4/cuda/11.0.1/cuda_nvcc/bin/nvcc
CXXFLAGS=--gpu-architecture=compute_75 \
--gpu-code=compute_75 \
-use_fast_math  \
-ftz=true \
--std=c++14 -O3 -g \
-Xcompiler=-march=native \
--compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0 \
-I/media/sdb4/cuda/11.0.1/cuda_cudart/targets/x86_64-linux/include/ 
#-I/media/sdb4/cuda/11.0.1/cuda_cudart/targets/x86_64-linux/include/cuda/std/detail/libcxx/include/


custd_00_cuda_main: custd_00_cuda_main.cu
	$(CXX) -o $@ $^ $(CXXFLAGS)
