g++ \
    lesson01.cpp -g -I/home/martin/src/Halide/b/include/ \
     -L /home/martin/src/Halide/b/src/ \
     -lHalide -lpthread -ldl -o lesson01 -std=c++20 \
     -Wl,-rpath=/home/martin/src/Halide/b/src/

# ./lesson01 -g my_first_generator -o . target=host-vulkan
# HL_JIT_TARGET=host-vulkan ./lesson01
