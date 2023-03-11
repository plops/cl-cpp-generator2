CXX=g++
CXXFLAGS=-std=c++17 -Wall -Wextra -Wpedantic $(shell pkg-config shaderc vulkan --cflags)
LIBS=$(shell pkg-config shaderc vulkan --libs)

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f main
