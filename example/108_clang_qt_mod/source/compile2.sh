clang++ \
    use.cpp -o use \
    -std=c++20 -ggdb -O1 \
    -fmodule-file=std_mod.pcm \
    `pkg-config Qt5Gui Qt5Widgets --cflags --libs`
