CXX = $(shell wx-config --cxx)

all: vis


CXXFLAGS=-Og -g3 -ggdb \
-Wall -Wextra -Werror \
-Wshadow -Wdouble-promotion -fno-common -Wconversion  \
-ffunction-sections -fdata-sections \
-Wno-error=inaccessible-base \
-Wno-error=unused-variable \
-Wno-error=deprecated-copy \
-Wno-error=float-conversion \
-Wno-error=shadow \
-Wno-error=conversion \
-Wno-error=double-promotion \
-pipe \
`wx-config --cxxflags` \
-march=native -mtune=native 

LDFLAGS= \
`wx-config --libs` \
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

