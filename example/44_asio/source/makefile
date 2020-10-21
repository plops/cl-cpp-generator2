all: vis

#CXX=g++-10.2.0
# -Og -g3 -ggdb 
# -Wno-error=inaccessible-base
# -Wshadow -Wdouble-promotion -fno-common -Wconversion 
CXXFLAGS=-O2 \
-std=gnu++17 \
-Wall -Wextra -Werror -Wfatal-errors\
-ffunction-sections -fdata-sections \
-Wno-error=unused-variable \
-Wno-error=unused-parameter \
-Wno-error=double-promotion \
-pipe \
-march=native -mtune=native \
-DASIO_STANDALONE

LDFLAGS= \
-L/usr/local/lib \
-Wl,--gc-sections \
-Wl,--print-gc-sections \
-lpthread

CFILES:=simple_00_client.cpp
#$(shell find simple*.| grep .cpp$)
OBJ:=$(CFILES:%.cpp=%.o)

all: server client

%.o: %.cpp globals.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

vis: $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

server: simple_01_server.o
	$(CXX) $^ -o $@ $(LDFLAGS)

client: simple_00_client.o
	$(CXX) $^ -o $@ $(LDFLAGS)

clean:
	rm vis *.o
