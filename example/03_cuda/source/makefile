CC=/opt/cuda/bin/nvcc
LDFLAGS=-lglfw -lGL -ldl -lGLU
CFLAGS=-IGL/include --std=c++14 -O3 -g -Xcompiler=-march=native -Xcompiler=-ggdb #-Xcompiler=-fsanitize=address

all: 03matmul 04matmul 06interop

GL/src/glad.o: GL/src/glad.c
	gcc -O3 -ggdb -march=native -ffast-math -c -o GL/src/glad.o -IGL/include   GL/src/glad.c #-fsanitize=address

06interop: 06interop.cu GL/src/glad.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ GL/src/glad.o $< 


%: %.cu
	$(CC) -o $@ $< 

clean:
	rm 03matmul 04matmul 06interop  GL/src/glad.o
