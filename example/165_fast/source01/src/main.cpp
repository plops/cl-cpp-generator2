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
  void operator[](size_t i) {
    return *mChunks[(i / ChunkSize)][(i % ChunkSize)];
  }
  void push_back(T &&value) { *mChunks.push_back(value); }

private:
  // similar to std::deque but that doesn't have a configurable chunk size,
  // which is usually chosen too small by the compiler
  using Chunk = boost::container::static_vector<T, ChunkSize>;
  std::vector<std::unique_ptr<Chunk>> mChunks;
};

void BM_StableVector() {
  stable_vector<int, 4 * 4096> v;
  std::list<int> tmp;
  for (decltype(0 + 100'000 + 1) i = 0; i < 100'000; i += 1) {
    // randomize heap
    for (decltype(0 + 1000 + 1) x = 0; x < 1000; x += 1) {
      tmp.push_back(x);
    }
  }
}

int main() {
  boost::multi_array<float, 3> a;
  for (decltype(0 + 10 + 1) i = 0; i < 10; i += 1) {
    a[i][i][i] = i;
  }
  stable_vector<float, 1024> mFloats;
  std::unordered_map<int, float *> mInstruments;
  BM_StableVector();
  // Working set size (WSS) is the memory you work with, not how much memory you
  // allocated or mapped. Measured in cache lines or pages (Brendan Gregg WSS
  // estimation tool wss.pl)
}
