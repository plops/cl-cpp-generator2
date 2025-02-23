#include <benchmark/benchmark.h>
#include <random>
#include <vector>

std::vector<uint64_t> GenerateShuffledIndices(uint64_t n) {
  std::mt19937 gen(std::random_device{}());
  std::vector<uint64_t> v(n);
  std::iota(v.begin(), v.end(), 0);
  std::shuffle(v.begin(), v.end(), gen);
  return v;
}

uint64_t Sum(std::vector<uint64_t> const &v, size_t n) {
  uint64_t sum{0};
  for (decltype(0 + n + 1) pos = 0; pos < n; pos += 1) {
    sum += v[v[pos]];
  }
  return sum;
}

void BM_Walk(benchmark::State &state) {
  auto kbytes{static_cast<size_t>(state.range(0))};
  auto n{(kbytes * 1024) / sizeof(uint64_t)};
  auto v{GenerateShuffledIndices(n)};
  for (auto &&_ : state) {
    auto sum{Sum(v, n)};
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(n * sizeof(uint64_t) * state.iterations());
}

BENCHMARK(BM_Walk)->RangeMultiplier(2)->Range(8, 8 << 13);
BENCHMARK_MAIN();