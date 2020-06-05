#CXX=x86_64-w64-mingw32-g++
#CC=x86_64-w64-mingw32-gcc

LDFLAGS = -lglfw -lGL

LDFLAGS_LUA = -ldl -lreadline -lm

# mostly static
#LDFLAGS=-L/usr/x86_64-w64-mingw32/usr/lib \
#-l:libglfw3.a \
#-lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32 -loleaut32 -luuid -lcomdlg32 -ladvapi32 \
#-lopengl32 \
#-lws2_32 \
#-static-libgcc \
#-static-libstdc++



#OPT=-Og -ggdb
#OPT=-Ofast -march=core2 -mtune=core2 -flto
OPT=-Ofast -march=native -mtune=native

#-flto

CXXFLAGS = \
$(OPT) \
-std=gnu++20 

CFLAGS= \
$(OPT) \
-Ilua \
-std=c99 \
-DLUA_USE_LINUX -DLUA_USE_READLINE

CFILES:=$(shell find .| grep .cpp$)
CUFILES:=$(shell ls | grep .cu$)
OBJ:=$(CFILES:%.cpp=%.o)
CUOBJ:=$(CUFILES:%.cu=%.o)
HFILES:=$(CFILES:%.cpp=%.h)

%.o: %.cpp globals.h
	$(CXX) -c -o $@ $< $(CXXFLAGS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS) 

%.o: %.cu
	/opt/cuda/bin/nvcc -c --std=c++14 -O3 -g -Xcompiler=-march=native -Xcompiler=-ggdb  --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.3.0 -o $@ $< 


all: vis

#all: vis_finisar_noadq

#vis_demo itla_ring_effect_photonics.zip



OBJ2:= \
vis_00_main.o \
vis_01_glfw_window.o \
vis_02_draw.o \
vis_03_gui.o \
vis_04_lua.o \
imgui/imgui_demo.o \
imgui/imgui_draw.o \
imgui/imgui_impl_glfw.o \
imgui/imgui_impl_opengl2.o \
imgui/imgui.o \
imgui/imgui_widgets.o


# martin@labolg ~/stage/py_wavelength_tune/test_lua_cpp/source $ find lua -type f -printf "\\n%p \\"|grep \\.c
CFILES_LUA:= lua/lcode.c \
lua/lstring.c \
lua/lcorolib.c \
lua/lundump.c \
lua/ltests.c \
lua/ltablib.c \
lua/lzio.c \
lua/lmathlib.c \
lua/llex.c \
lua/lbaselib.c \
lua/lapi.c \
lua/lauxlib.c \
lua/testes/libs/lib2.c \
lua/testes/libs/lib22.c \
lua/testes/libs/lib21.c \
lua/testes/libs/lib1.c \
lua/testes/libs/lib11.c \
lua/lstrlib.c \
lua/lobject.c \
lua/lua.c \
lua/ldebug.c \
lua/loadlib.c \
lua/lparser.c \
lua/linit.c \
lua/ltm.c \
lua/loslib.c \
lua/lopcodes.c \
lua/lgc.c \
lua/ldump.c \
lua/onelua.c \
lua/lvm.c \
lua/lutf8lib.c \
lua/lmem.c \
lua/lctype.c \
lua/lstate.c \
lua/ldo.c \
lua/lfunc.c \
lua/ldblib.c \
lua/ltable.c \
lua/liolib.c

OBJ_LUA:=$(CFILES_LUA:%.c=%.o)


CORE_O= lapi.o lcode.o lctype.o ldebug.o ldo.o ldump.o lfunc.o lgc.o llex.o \
        lmem.o lobject.o lopcodes.o lparser.o lstate.o lstring.o ltable.o \
        ltm.o lundump.o lvm.o lzio.o ltests.o
LIB_O=  lbaselib.o ldblib.o liolib.o lmathlib.o loslib.o ltablib.o lstrlib.o \
        lutf8lib.o loadlib.o lcorolib.o linit.o




vis:   lua/lauxlib.o  $(LIB_O:%.o=lua/%.o) $(CORE_O:%.o=lua/%.o) $(OBJ2)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(LDFLAGS_LUA)

genclean:
	rm *.cpp *.o globals.h utils.h proto2.h vis *.s imgui/*.o  SPinGW/*.o lua/*.o

clean:
	rm *.o vis vis.exe *.s imgui/*.o lua/*.o
