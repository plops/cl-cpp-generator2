CXX = clang++
CXXFLAGS=-std=gnu++2b \
 `pkg-config libpipewire-0.3 --cflags` \
 `pkg-config fmt --cflags` \
  -ggdb3 -O0

LDFLAGS=`pkg-config libpipewire-0.3 --libs` \
        `pkg-config fmt --libs` \


all: main

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c

main:  main.o
	$(CXX)  -o main $^  $(LDFLAGS)

clean:
	rm *.o main
