CC=/opt/cuda/bin/nvcc

all: 03matmul 04matmul

%: %.cu
	$(CC) -o $@ $<

clean:
	rm 03matmul 04matmul
