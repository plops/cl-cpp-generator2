#include "papipp.h"
#include <benchmark/benchmark.h>
#include <iostream>
#include <list>
#include <map>
#include <stable_vector.h>
#include <unordered_map>
#include <vector>
template <typename T> int Sum(T v) {
  int sum{0};
  for (decltype(0 + v.size() + 1) i = 0; i < v.size(); i += 1) {
    sum += v[i];
  }
  return sum;
}

void BM_StableVector(benchmark::State &state) {
  stable_vector<int, 4 * 4096> v;
  std::list<int> tmp;
  papi::event_set<PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_BR_MSP, PAPI_L1_DCM> events;
  events.start_counters();
  for (decltype(0 + 100'000 + 1) i = 0; i < 100'000; i += 1) {
    // randomize heap by filling list (this makes the micro-benchmark more like
    // the real thing)
    for (decltype(0 + 1000 + 1) x = 0; x < 1000; x += 1) {
      tmp.push_back(x);
    }
    v.push_back(i);
  }
  for (auto &&_ : state) {
    auto sum{Sum<stable_vector<int, 4 * 4096>>(v)};
    benchmark::DoNotOptimize(sum);
  }
  events.stop_counters();
  (std::cout << std::format("(:events.get<PAPI_TOT_INS>().counter() '{}')\n",
                            events.get<PAPI_TOT_INS>().counter()))(
      std::cout << std::format("(:events.get<PAPI_TOT_CYC>().counter() '{}')\n",
                               events.get<PAPI_TOT_CYC>().counter()),
      std::cout << std::format("(:events.get<PAPI_BR_MSP>().counter() '{}')\n",
                               events.get<PAPI_BR_MSP>().counter()),
      std::cout << std::format("(:events.get<PAPI_L1_DCM>().counter() '{}')\n",
                               events.get<PAPI_L1_DCM>().counter()));
}

void BM_StableVectorReserved(benchmark::State &state) {
  stable_vector<int, 4 * 4096> v;
  v.reserve(100'000);
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
    auto sum{Sum<stable_vector<int, 4 * 4096>>(v)};
    benchmark::DoNotOptimize(sum);
  }
}

void BM_Vector(benchmark::State &state) {
  std::vector<int> v;
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
    auto sum{Sum<std::vector<int>>(v)};
    benchmark::DoNotOptimize(sum);
  }
}

void BM_VectorReserved(benchmark::State &state) {
  std::vector<int> v;
  v.reserve(100'000);
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
    auto sum{Sum<std::vector<int>>(v)};
    benchmark::DoNotOptimize(sum);
  }
}

void BM_UnorderedMap(benchmark::State &state) {
  std::unordered_map<int, int> v;
  std::list<int> tmp;
  for (decltype(0 + 100'000 + 1) i = 0; i < 100'000; i += 1) {
    // randomize heap by filling list (this makes the micro-benchmark more like
    // the real thing)
    for (decltype(0 + 1000 + 1) x = 0; x < 1000; x += 1) {
      tmp.push_back(x);
    }
    v.emplace(i, i);
  }
  for (auto &&_ : state) {
    auto sum{Sum<std::unordered_map<int, int>>(v)};
    benchmark::DoNotOptimize(sum);
  }
}

void BM_UnorderedMapReserved(benchmark::State &state) {
  std::unordered_map<int, int> v;
  v.reserve(200'000);
  std::list<int> tmp;
  for (decltype(0 + 100'000 + 1) i = 0; i < 100'000; i += 1) {
    // randomize heap by filling list (this makes the micro-benchmark more like
    // the real thing)
    for (decltype(0 + 1000 + 1) x = 0; x < 1000; x += 1) {
      tmp.push_back(x);
    }
    v.emplace(i, i);
  }
  for (auto &&_ : state) {
    auto sum{Sum<std::unordered_map<int, int>>(v)};
    benchmark::DoNotOptimize(sum);
  }
}

void BM_Map(benchmark::State &state) {
  std::map<int, int> v;
  std::list<int> tmp;
  for (decltype(0 + 100'000 + 1) i = 0; i < 100'000; i += 1) {
    // randomize heap by filling list (this makes the micro-benchmark more like
    // the real thing)
    for (decltype(0 + 1000 + 1) x = 0; x < 1000; x += 1) {
      tmp.push_back(x);
    }
    v.emplace(i, i);
  }
  for (auto &&_ : state) {
    auto sum{Sum<std::map<int, int>>(v)};
    benchmark::DoNotOptimize(sum);
  }
}

BENCHMARK(BM_StableVector);
BENCHMARK(BM_StableVectorReserved);
BENCHMARK(BM_Vector);
BENCHMARK(BM_VectorReserved);
BENCHMARK(BM_UnorderedMap);
BENCHMARK(BM_UnorderedMapReserved);
BENCHMARK(BM_Map);
BENCHMARK_MAIN();