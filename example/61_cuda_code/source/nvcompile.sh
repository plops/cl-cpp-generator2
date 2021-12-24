clang++ -x cuda --cuda-path=/usr/local/cuda \
-I. \
-L/usr/local/cuda/targets/x86_64-linux/lib/ \
-lcudart -ldl -lrt -pthread \
--cuda-gpu-arch=sm_75 \
-ffp-contract=fast \
-ffast-math quadtree.cu
