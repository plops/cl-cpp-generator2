CXX = clang++
CXXFLAGS=-std=gnu++2b \
 `sdl2-config --cflags` \
 -ggdb3 -O1

LDFLAGS=`sdl2-config --libs`

all: main

%.o: %.cpp FancyWindow.h c_resource.hpp
	$(CXX) $(CXXFLAGS) $< -c

main:  FancyWindow.o main.o 
	$(CXX)  -o main $^  $(LDFLAGS)

clean:
	rm main.o FancyWindow.o main
