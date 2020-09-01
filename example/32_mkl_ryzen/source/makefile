all: libfake_intel.so

libfake_intel.so: vis_00_fake_intel.cpp
	g++ -shared -fPIC -O2 -march=native -o $@ $^

dgemm: vis_01_base.cpp
	g++ `pkg-config mkl-static-lp64-iomp --cflags --libs` -o $@ $^
