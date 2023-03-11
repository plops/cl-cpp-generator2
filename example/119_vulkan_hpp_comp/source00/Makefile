CXX=g++
CXXFLAGS=-ggdb -O0 -std=c++17 -Wall -Wextra -Wpedantic $(shell pkg-config shaderc vulkan --cflags)
LIBS=$(shell pkg-config shaderc vulkan --libs)
all: main
main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f main
