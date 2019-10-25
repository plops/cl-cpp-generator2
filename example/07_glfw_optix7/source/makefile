CC=gcc
LDFLAGS=`pkg-config --static --libs glfw3` -lGL
CFLAGS=-march=native -ffast-math -std=gnu18 -O0 -ggdb -fsanitize=address  #`pkg-config --cflags cglm`  
#CFLAGS=-std=c18 -O2 -march=native
#CFLAGS=-std=c18 -O0 -Wno-variadic-macros -W -Wpedantic -Wextra -Wall -Wcast-align -Wcast-qual -Wstrict-aliasing=2 -Wframe-larger-than=32768 -Wno-strict-overflow -Wsync-nand -Wtrampolines -Wsign-compare -Werror=float-equal -Werror=missing-braces -Werror=init-self -Werror=logical-op -Werror=write-strings -Werror=address -Werror=array-bounds -Werror=char-subscripts -Werror=enum-compare -Werror=implicit-int -Werror=empty-body -Werror=main -Werror=aggressive-loop-optimizations -Werror=nonnull -Werror=parentheses -Werror=pointer-sign -Werror=return-type -Werror=sequence-point -Werror=uninitialized -Werror=volatile-register-var -Werror=ignored-qualifiers -Werror=missing-parameter-type -Werror=old-style-declaration -Wno-error=maybe-uninitialized -Wno-unused-function -Wodr -Wformat-signedness -Wsuggest-final-types -Wsuggest-final-methods -Wno-ignored-attributes -Wno-missing-field-initializers -Wshift-overflow=2 -Wduplicated-cond -Wduplicated-branches -Werror=restrict -Wdouble-promotion -Wformat=2
CFILES:=$(shell ls | grep .c$)
OBJ:=$(CFILES:%.c=%.o)
HFILES:=$(CFILES:%.c=%.h)
PROTOS:=$(patsubst %, proto_%, $(HFILES))
%.o: %.c #proto2.h
	$(CC) -c -o $@ $< $(CFLAGS)

all: optix_demo
optix_demo: $(OBJ) 
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

genclean:
	rm *.c *.o optix*.h proto_*.h globals.h utils.h proto2.h optix_demo *.s

clean:
	rm *.o optix*.h proto_*.h optix_demo *.s

proto2.h: $(PROTOS)
	echo '#ifndef PROTO2_H' > $@
	echo '#define PROTO2_H' >> $@
	cat $^ >> $@
	echo '#endif' >> $@

proto_%.h: %.c
	sh ./gen_proto2.sh $<
