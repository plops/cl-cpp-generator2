# Trying out IGL

- it is a library to render realtime


## compile the library

```
cd ~/src
git clone https://github.com/facebook/igl
# this downloads 25MB 

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
# this downloads some more dependencies:
# meshoptimizer
# glslang

```

## first example

- go through
  https://github.com/facebook/igl/blob/main/samples/desktop/Tiny/Tiny.cpp
