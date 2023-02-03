time clang++ -std=c++20 -ggdb -O1 -fmodule-file=fatheader.pcm main.cpp -c -o main.o
time clang++ -std=c++20 -ggdb -O1 main.o -o main  `pkg-config Qt5Gui Qt5Widgets --libs`
