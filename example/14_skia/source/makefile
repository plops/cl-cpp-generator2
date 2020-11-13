all: vis

#CXX=g++-10.2.0
# -Og -g3 -ggdb 
# -Wno-error=inaccessible-base
# -Wshadow -Wdouble-promotion -fno-common -Wconversion 
CXXFLAGS=-O2 \
-std=gnu++2a \
-Wall -Wextra -Werror \
-ffunction-sections -fdata-sections \
-Wno-error=unused-variable \
-Wno-error=unused-parameter \
-Wno-error=double-promotion \
-pipe \
-march=native -mtune=native \
-I/home/martin/src/skia/

LDFLAGS= \
-L/home/martin/src/skia/out/Shared \
-lskia -lSDL2 -lGL \
-Wl,--gc-sections \
-Wl,--print-gc-sections \
-Wl,--rpath=/home/martin/src/skia/out/Shared

CFILES:=$(shell find .| grep .cpp$)
OBJ:=$(CFILES:%.cpp=%.o)

all: vis

%.o: %.cpp globals.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

vis: $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	rm vis *.o
