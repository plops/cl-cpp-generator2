CXX=g++
LDFLAGS=
CXXFLAGS=-march=native -ffast-math -std=gnu++1z -O0 -ggdb -fsanitize=address  #`pkg-config --cflags cglm`  
CFILES:=$(shell ls | grep .cpp$)
OBJ:=$(CFILES:%.c=%.o)
HFILES:=$(CFILES:%.c=%.h)
PROTOS:=$(patsubst %, proto_%, $(HFILES))
%.o: %.cpp #proto2.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

all: copernicus
copernicus: $(OBJ) 
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

genclean:
	rm *.cpp *.o proto_*.h globals.h utils.h proto2.h copernicus *.s

clean:
	rm *.o proto_*.h copernicus *.s

proto2.h: $(PROTOS)
	echo '#ifndef PROTO2_H' > $@
	echo '#define PROTO2_H' >> $@
	cat $^ >> $@
	echo '#endif' >> $@

proto_%.h: %.c
	sh ./gen_proto2.sh $<
