#CXX=x86_64-w64-mingw32-g++
#CC=x86_64-w64-mingw32-gcc

LDFLAGS = -ltbb
#OPT=-O0
#OPT=-Og -ggdb -march=native
#OPT=-Ofast -march=core2 -mtune=core2 -flto
OPT=-Ofast -march=native -mtune=native -flto

#-flto

CXXFLAGS = \
$(OPT) \
-std=gnu++20
#-fcoroutines

CFLAGS= \
$(OPT) \
-std=c99 

CFILES:=$(shell find .| grep .cpp$)
CUFILES:=$(shell ls | grep .cu$)
OBJ:=$(CFILES:%.cpp=%.o)
CUOBJ:=$(CUFILES:%.cu=%.o)

%.o: %.cpp globals.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS) 

%.o: %.cu
	/opt/cuda/bin/nvcc -c --std=c++14 -O3 -g -Xcompiler=-march=native -Xcompiler=-ggdb  --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.3.0 -o $@ $< 


all: vis

#all: vis_finisar_noadq

#vis_demo itla_ring_effect_photonics.zip



OBJ2:= \
vis_00_main.o 

vis:  $(OBJ2)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

clean:
	rm *.o vis vis.exe *.s 

