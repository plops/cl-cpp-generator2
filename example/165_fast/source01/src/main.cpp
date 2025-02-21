#include <boost/container/static_vector.hpp>
#include <memory>
#include <vector>
template <class T, size_t ChunkSize> class stable_vector {
  static_assert(0 == (ChunkSize % 2), "ChunkSize needs to be a multiple of 2");
  void operator[](size_t i) {
    return *mChunks[(i / ChunkSize)][(i % ChunkSize)];
  }
  using Chunk = boost::container::static_vector<T, ChunkSize>;
  std::vector<std::unique_ptr<Chunk>> mChunks;
};

int main() {}
