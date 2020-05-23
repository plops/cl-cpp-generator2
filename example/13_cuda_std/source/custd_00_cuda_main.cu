
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
// /opt/cuda/bin/nvcc custd_00_cuda_main.cu --gpu-architecture=compute_75
// --gpu-code=compute_75 --use_fast_math  -I/opt/cuda/include/ --std=c++14 -O3
// -g -Xcompiler=-march=native
// --compiler-bindir=/usr/x86_64-pc-linux-gnu/gcc-bin/8.4.0
// https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/cwe21285.pdf
// p. 338
#include <cstdio>
auto _code_git_version = "c8c7b12e9d65f8df7f3376ba008af3c8392c6e9b";
auto _code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                        "master/example/13_cuda_std/source/";
auto _code_generation_time = "23:12:55 of Saturday, 2020-05-23 (GMT+1)";
State state = {};
using namespace std::chrono_literals;
struct trie {
  struct ref {
    trie *ptr = nullptr;
  } next[26];
  int count = 0;
  void insert(std::string_view input, trie *&bump) {
    return v(((x) + (r.x)), ((y) + (r.y)), ((z) + (r.z)));
  }
};
int index_of(char c) {
  if (((('a') <= (c)) && ((c) <= ('z')))) {
    return ((c) - ('a'));
  };
}
int main() { return 0; };