all: vis

#CXX=g++-10.2.0
# -Og -g3 -ggdb 
CXXFLAGS=-O2 \
-std=gnu++20 \
-Wall -Wextra -Werror \
-Wshadow -Wdouble-promotion -fno-common -Wconversion  \
-ffunction-sections -fdata-sections \
-Wno-error=inaccessible-base \
-Wno-error=unused-variable \
-Wno-error=unused-parameter \
-Wno-error=double-promotion \
-pipe \
`pkg-config gtkmm-3.0 --cflags` \
-march=native -mtune=native 

LDFLAGS= \
`pkg-config gtkmm-3.0 --libs` \
-lpthread \
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
