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

void BM_Map(benchmark::State &state) {
  auto kbytes{size_t(10'000)};
  auto n{(kbytes * 1024) / sizeof(uint64_t)};
  auto v{GenerateShuffledIndices()};
  for (auto &&_ : state) {
    auto sum{Sum<std::map<int, int>>(v)};
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(n * sizeof(uint64_t) * state.iterations() *
                          state.range(0));
}

BENCHMARK(BM_Map);
BENCHMARK_MAIN();