quadtree.o: quadtree.cu quadtree.h
	nvcc -ccbin g++ -m64 -I.  -dc --std=c++14 --threads 0 -gencode arch=compute_75,code=compute_75 -o quadtree.o quadtree.cu

quadtree: quadtree.o
	nvcc -ccbin g++ -m64 -gencode arch=compute_75,code=sm_75 -o quadtree quadtree.o -lcudadevrt
