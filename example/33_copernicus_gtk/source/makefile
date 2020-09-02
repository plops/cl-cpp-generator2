# https://interrupt.memfault.com/blog/best-and-worst-gcc-clang-compiler-flags
# -Wundef .. not working with gtkmm
# ;-g3 -ggdb 
CXXFLAGS=-Os -g3 -ggdb \
-Wall -Wextra -Werror \
-Wshadow -Wdouble-promotion -fno-common -Wconversion  \
-ffunction-sections -fdata-sections \
-Wno-error=unused-variable \
-pipe \
`pkg-config gtkmm-3.0 --cflags` \
-march=native -mtune=native 

LDFLAGS= \
`pkg-config gtkmm-3.0 --libs` \
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
