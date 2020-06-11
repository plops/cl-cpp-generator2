CXX=/media/sdb4/cuda/b/cuda_nvcc/bin/nvcc
CXXFLAGS=--gpu-architecture=compute_75 \
--gpu-code=compute_75 \
--use_fast_math  \
--std=c++14 -O3 -g \
-Xcompiler=-march=native \
--compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0 \
-I/media/sdb4/cuda/b/cuda_cudart/targets/x86_64-linux/include/


custd_00_cuda_main: custd_00_cuda_main.cu
	$(CXX) -o $@ $^ $(CXXFLAGS)
