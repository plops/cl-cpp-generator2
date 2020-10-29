all: vis

#CXX=g++-10.2.0
# -Og -g3 -ggdb 
# -Wno-error=inaccessible-base
# -Wshadow -Wdouble-promotion -fno-common -Wconversion 
CXXFLAGS=-O1 -ffast-math \
-std=gnu++17 \
-Wall -Wextra -Werror -Wfatal-errors\
-ffunction-sections -fdata-sections \
-Wno-error=unused-variable \
-Wno-error=unused-parameter \
-Wno-error=double-promotion \
-pipe \
-march=native -mtune=native \
-I. 

LDFLAGS= \
-L/usr/local/lib

#-Wl,--gc-sections 
#-Wl,--print-gc-sections

CFILES:=$(shell find *.cpp| grep .cpp$)
OBJ:=$(CFILES:%.cpp=%.o)

all: vis vis1 run vis2 vispg runpg

%.o: %.cpp globals.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

vis: $(OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

vis1: vis_00_base.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ -fprofile-generate=./prof_gmon.out $(LDFLAGS) -lgcov

run: vis1
	./vis1 > vis1.out

vis2: vis_00_base.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ -fprofile-use=./prof_gmon.out $(LDFLAGS) 

vispg: vis_00_base.cpp
	$(CXX) -pg $(CXXFLAGS) $^ -o $@  $(LDFLAGS) -lgcov -fprofile-use=./prof_gmon.out

runpg: vispg
	./vispg > vispg.out
	gprof vispg gmon.out > vispg_analysis.txt

clean:
	rm -rf gmon.out vis *.o vis1 vis2 vispg prof_gmon.out vis1.out vispg.out
