CXX=g++
LDFLAGS=-lglfw -lGL -L/opt/cuda/lib64 -lcudart -lcufft -lcufftw
CXXFLAGS=-march=native -ffast-math -std=gnu++1z -O2  -ggdb #-pg # -fsanitize=address  #`pkg-config --cflags cglm` 
CFILES:=$(shell find .| grep .cpp$)
CUFILES:=$(shell ls | grep .cu$)
OBJ:=$(CFILES:%.cpp=%.o)
CUOBJ:=$(CUFILES:%.cu=%.o)
HFILES:=$(CFILES:%.cpp=%.h)
PROTOS:=$(patsubst %, proto_%, $(HFILES))
%.o: %.cpp #proto2.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

%.o: %.cu
	/opt/cuda/bin/nvcc -c --std=c++14 -O3 -g -Xcompiler=-march=native -Xcompiler=-ggdb  --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.3.0 -o $@ $< 


all: doppler
doppler: $(OBJ) $(CUOBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)



genclean:
	rm *.cpp *.o globals.h utils.h proto2.h doppler *.s

clean:
	rm *.o doppler *.s

