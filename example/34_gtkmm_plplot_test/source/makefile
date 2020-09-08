all: vis

#vis: *.cpp
#	g++ $^ -o $@ `pkg-config gtkmm-3.0 --cflags --libs` `pkg-config --cflags --libs gtkmm-plplot-2.0`


CXXFLAGS=-Og -g3 -ggdb \
-Wall -Wextra -Werror \
-Wshadow -Wdouble-promotion -fno-common -Wconversion  \
-ffunction-sections -fdata-sections \
-Wno-error=inaccessible-base \
-Wno-error=unused-variable \
-pipe \
`pkg-config gtkmm-3.0 --cflags` \
`pkg-config --cflags gtkmm-plplot-2.0` \
-march=native -mtune=native 

LDFLAGS= \
`pkg-config gtkmm-3.0 --libs` \
`pkg-config --libs gtkmm-plplot-2.0` \
-Wl,--gc-sections \
-Wl,--print-gc-sections

CFILES:=$(shell find .| grep .cpp$)
OBJ:=$(CFILES:%.cpp=%.o)

all: vis

%.o: %.cpp globals.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

vis: $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	rm vis *.o
