#include <vector>
template <class T, size_t ChunkSize> void stable_vector::operator[](size_t i) {
  return *mChunks[(i / ChunkSize)][(i % ChunkSize)];
};

int main() {}
