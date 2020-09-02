CXXFLAGS=-O2 \
-pipe \
`pkg-config gtkmm-3.0 --cflags --libs` \
-march=native -mtune=native 

all: vis

vis: *.cpp
	g++ $^ -o $@ $(CXXFLAGS)

clean:
	rm vis
