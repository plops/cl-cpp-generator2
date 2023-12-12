#g++ --std=c++20 -fmodules-ts -fpermissive Vector.cpp -c
#g++ --std=c++20 -fmodules-ts -fpermissive user.cpp -c

rm -rf gcm.cache/
#gcc -std=c++20 -fmodules-ts -x c++-system-header cstdio -x c++ Vector.cpp user.cpp -o main
gcc -std=c++20 -fmodules-ts -fpermissive Vector.cpp user.cpp -o main -lm
