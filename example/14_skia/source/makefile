all: vis

CXX=clang++
# -Og -g3 -ggdb 
# -Wno-error=inaccessible-base
# -Wshadow -Wdouble-promotion -fno-common -Wconversion 
# -Werror
# gnu++2a
CXXFLAGS=-Og -g3 -ggdb \
-std=gnu++2a  \
-Wall -Wextra \
-ffunction-sections -fdata-sections \
-Wno-error=unused-variable \
-Wno-error=unused-parameter \
-Wno-error=double-promotion \
-pipe \
-march=native -mtune=native \
-I/home/martin/src/skia/

SKIA=/home/martin/src/skia/out/Release-x64
#SKIA=/home/martin/src/skia/out/Release_lowdeps

LDFLAGS= \
-L$(SKIA) \
-lskia \
-lSDL2 -lGL \
-Wl,--gc-sections \
-Wl,--print-gc-sections \
-Wl,--rpath=$(SKIA)

#-lskottie -lskparagraph -lsksg -lskshaper \
#-ldng_sdk -lparticles -lpiex -lsfntly -lskresources -lspvtools -lspvtools_val \
#

CFILES:=$(shell find .| grep .cpp$)
OBJ:=$(CFILES:%.cpp=%.o)

all: vis

%.o: %.cpp globals.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

vis: $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	rm vis *.o
