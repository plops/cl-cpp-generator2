time clang++ \
     use.cpp -c -o use.o \
     -std=c++20 -ggdb -O1 \
     -fmodule-file=std_mod.pcm

# 8.9sec

time clang++ use.o -o use \
     `pkg-config Qt5Gui Qt5Widgets --cflags --libs`

# 0.5sec
