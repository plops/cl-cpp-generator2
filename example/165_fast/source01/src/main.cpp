#include <boost/container/static_vector.hpp>
#include <memory>
#include <unordered_map>
#include <vector>
// simple container that keeps things together
template <class T, size_t ChunkSize> class stable_vector {
  static_assert(0 == (ChunkSize % 2), "ChunkSize needs to be a multiple of 2");
  void operator[](size_t i) {
    return *mChunks[(i / ChunkSize)][(i % ChunkSize)];
  }
  // similar to std::deque but that doesn't have a configurable chunk size,
  // which is usually chosen too small by the compiler
  using Chunk = boost::container::static_vector<T, ChunkSize>;
  std::vector<std::unique_ptr<Chunk>> mChunks;
};

int main() {}

stable_vector<float, 1024> mFloats;
std::unordered_map<int, float *> mInstruments;
// Working set size (WSS) is the memory you work with, not how much memory you
// allocated or mapped. Measured in cache lines or pages