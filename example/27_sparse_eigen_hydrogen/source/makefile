CXX=/media/sdb4/cuda/11.0.1/cuda_nvcc/bin/nvcc
CXXFLAGS=--gpu-architecture=compute_75 \
--gpu-code=compute_75 \
-use_fast_math  \
-ftz=true \
--std=c++17 -O1 -g \
-Xcompiler=-march=native \
--compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0 \
-I/media/sdb4/cuda/11.0.1/include \
-Iarpack-ng/b/
# -I/media/sdb4/cuda/11.0.1/cuda_cudart/targets/x86_64-linux/include/ 
#-I/media/sdb4/cuda/11.0.1/cuda_cudart/targets/x86_64-linux/include/cuda/std/detail/libcxx/include/
LDFLAGS=-L/media/sdb4/cuda/11.0.1/lib \
-L/media/sdb4/cuda/11.0.1/cuda_cudart/targets/x86_64-linux/lib \
-Larpack-ng/b/ -larpack \
-llapack \
-lblas 

%.o: %.cu
	$(CXX) -c -o $@ $^ $(CXXFLAGS)

vis_00_cuda_main: vis_00_cuda_main.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm *.o vis_00_cuda_main
