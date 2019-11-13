CC=/opt/cuda/bin/nvcc
LDFLAGS=-lglfw -lGL -ldl
CFLAGS=-IGL/include  --std=c++14 -O3 -g -Xcompiler=-march=native -Xcompiler=-ggdb

all: 03matmul 04matmul 06interop

06interop: 06interop.cu GL/src/glad.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ GL/src/glad.c $< 


%: %.cu
	$(CC) -o $@ $< 

clean:
	rm 03matmul 04matmul 06interop
