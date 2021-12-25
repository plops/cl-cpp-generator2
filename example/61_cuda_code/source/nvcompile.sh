/usr/local/cuda/bin/nvcc \
-I. \
 --gpu-architecture=sm_75 \
 quadtree.cu

 #-dlto -arch=sm_75

# /usr/local/cuda/bin/nvcc -ccbin g++ -I. quadtree.cu -dc --std=c++14 --threads 0 -gencode arch=compute_75,code=compute_75

# /usr/local/cuda/bin/nvcc -ccbin g++ -I../../Common -m64 -dc --std=c++14 --threads 0 -gencode arch=compute_75,code=sm_75 -o cdpQuadtree.o -c cdpQuadtree.cu
#/opt/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o cdpQuadtree cdpQuadtree.o -lcudadevrt
