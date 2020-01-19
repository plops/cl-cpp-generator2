CXX=g++
LDFLAGS=
CXXFLAGS=-march=native -ffast-math -std=gnu++1z -O2  -ggdb -fsanitize=address  
CFILES:=$(shell find .| grep .cpp$)
OBJ:=$(CFILES:%.cpp=%.o)

%.o: %.cpp #proto2.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

all: fsm
fsm: $(OBJ) 
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

genclean:
	rm *.cpp *.o globals.h utils.h proto2.h fsm *.s

clean:
	rm *.o fsm *.s

