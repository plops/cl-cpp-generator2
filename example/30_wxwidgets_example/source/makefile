CXX = $(shell wx-config --cxx)
LDFLAGS=`wx-config --libs`
CXXFLAGS=-march=native -O2 -mtune=native \
`wx-config --cxxflags`

PROGRAM = vis

OBJECTS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))

.SUFFIXES: .o .cpp

.cpp.o :
	$(CXX) -c $(CXXFLAGS) -o $@ $<

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS)
	$(CXX) -o $(PROGRAM) $(OBJECTS) $(LDFLAGS)

clean:
	rm -f *.o $(PROGRAM)
