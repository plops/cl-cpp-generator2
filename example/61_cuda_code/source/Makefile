all: quadtree

NVDEBUG=-g -G
NVCFLAGS=-m64 -dc --std=c++14 --threads 0 -I. ${NVDEBUG}
NVGEN=-gencode arch=compute_75,code=compute_75
NVLFLAGS=-m64 -lcudadevrt ${NVDEBUG}

quadtree.o: quadtree.cu quadtree.h
	nvcc -ccbin g++ ${NVCFLAGS} ${NVGEN} -o quadtree.o quadtree.cu

quadtree: quadtree.o
	nvcc -ccbin g++ ${NVGEN} -o quadtree quadtree.o ${NVLFLAGS}

clean:
	rm quadtree quadtree.o
