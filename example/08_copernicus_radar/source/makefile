CXX=g++
LDFLAGS=
CXXFLAGS=-march=native -ffast-math -std=gnu++1z -O2 # -ggdb -pg # -fsanitize=address  #`pkg-config --cflags cglm` 
CFILES:=$(shell ls | grep .cpp$)
OBJ:=$(CFILES:%.cpp=%.o)
HFILES:=$(CFILES:%.cpp=%.h)
PROTOS:=$(patsubst %, proto_%, $(HFILES))
%.o: %.cpp #proto2.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

all: copernicus
copernicus: $(OBJ) 
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

genclean:
	rm *.cpp *.o globals.h utils.h proto2.h copernicus *.s

clean:
	rm *.o copernicus *.s

