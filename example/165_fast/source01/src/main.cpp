#include <benchmark/benchmark.h>
#include <boost/container/devector.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/multi_array.hpp>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>
// simple container that keeps things together
template <class T, size_t ChunkSize> class stable_vector {
  static_assert(0 == (ChunkSize % 2), "ChunkSize needs to be a multiple of 2");

public:
  T &operator[](size_t index) {
    return (*mChunks[(index / ChunkSize)])[(index % ChunkSize)];
  }
  void push_back(T value) {
    mN++;
    *(mChunks.push_back(value));
  }
  size_t size() { return mN; }

private:
  // similar to std::deque but that doesn't have a configurable chunk size,
  // which is usually chosen too small by the compiler
  using Chunk = boost::container::static_vector<T, ChunkSize>;
  std::vector<std::unique_ptr<Chunk>> mChunks;
  size_t mN;
};

int Sum(stable_vector<int, 4 * 4096> v) {
  int sum{0};
  for (decltype(0 + v.size() + 1) i = 0; i < v.size(); i += 1) {
    sum += v[i];
  }
}

void BM_StableVector(benchmark::State &state) {
  stable_vector<int, 4 * 4096> v;
  std::list<int> tmp;
  for (decltype(0 + 100'000 + 1) i = 0; i < 100'000; i += 1) {
    // randomize heap by filling list (this makes the micro-benchmark more like
    // the real thing)
    for (decltype(0 + 1000 + 1) x = 0; x < 1000; x += 1) {
      tmp.push_back(x);
    }
    v.push_back(i);
  }
  for (auto &&_ : state) {
    auto sum{Sum(v)};
    benchmark::DoNotOptimize(sum);
  }
}

int main() {
  stable_vector<float, 1024> mFloats;
  std::unordered_map<int, float *> mInstruments;
  BENCHMARK(BM_StableVector);
  // Working set size (WSS) is the memory you work with, not how much memory you
  // allocated or mapped. Measured in cache lines or pages (Brendan Gregg WSS
  // estimation tool wss.pl)
}
