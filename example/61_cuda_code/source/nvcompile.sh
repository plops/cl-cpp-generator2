/usr/local/cuda/bin/nvcc \
-I. \
 --gpu-architecture=sm_75 \
 quadtree.cu

 #-dlto -arch=sm_75

#/usr/local/cuda/bin/nvcc --cbin g++ -I. quadtree.cu -dc --std=c++14 --threads 0 -gencode arch=compute_75,code=compute_75

#/usr/local/cuda/bin/nvcc -ccbin g++ -I../../Common -m64 -dc --std=c++14 --threads 0 -gencode arch=compute_75,code=sm_75 -o cdpQuadtree.o -c cdpQuadtree.cu
