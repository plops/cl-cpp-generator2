
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
// https://on-demand.gputechconf.com/supercomputing/2019/video/sc1942-the-cuda-c++-standard-library/
#include <cstdio>
#include <cuda/std/atomic>
auto _code_git_version = "bd4bb1625df2448d51539c8fb9f26a62983bfd82";
auto _code_repository = "https://github.com/plops/cl-cpp-generator2/tree/"
                        "master/example/13_cuda_std/source/";
auto _code_generation_time = "00:06:05 of Sunday, 2020-05-24 (GMT+1)";
State state = {};
using namespace std::chrono_literals;
struct trie {
  struct ref {
    cuda::std::atomic<trie *> ptr = ATOMIC_VAR_INIT(nullptr);
    cuda::std::atomic_flag flag = ATOMIC_FLAG_INIT;
  } next[26];
  cuda::std::atomic<int> count = ATOMIC_VAR_INIT(0);
  __host__ __device__ void insert(cuda::std::string_view input,
                                  cuda::std::atomic<trie *> &bump) {
    auto n = this;
    for (auto pc : input) {
      auto const index = index_of(pc);
      if ((index) == (-1)) {
        if ((n) != (this)) {
          (n->count)++;
          n = this;
        };
        continue;
      };
      if ((n->next[index].ptr) == (nullptr)) {
        (bump)++;
        n->next[index].ptr = bump;
      };
      n = n->next[index].ptr;
    };
  };
};
int index_of(char c) {
  if (((('a') <= (c)) && ((c) <= ('z')))) {
    return ((c) - ('a'));
  };
  if (((('A') <= (c)) && ((c) <= ('Z')))) {
    return ((c) - ('A'));
  };
  return -1;
}
int main() { return 0; };