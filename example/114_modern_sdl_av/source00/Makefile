CXX = clang++
CXXFLAGS=-std=c++20 \
 `sdl2-config --cflags` \
 -ggdb3 -O1

LDFLAGS=`sdl2-config --libs`

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c

main: main.o FancyWindow.o
	$(CXX)  -o main $<  $(LDFLAGS)
