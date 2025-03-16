//
// Created by martin on 3/16/25.
//

#ifndef HISTOGRAM_H
#define HISTOGRAM_H
#include <array>
#include <cstdint>
#include <limits>


template <typename T, int N>
class Histogram {
    using namespace std;

public:
    Histogram(T mi, T ma, T bins = 128) noexcept :
        binMin{mi}, binMax{ma}, binCount{bins} {
        assert(binMin < binMax);
        binY.fill(0);
        {
            auto count = 0;
            for (auto&& x : binX) {
                x = mi + count * (mi - mi) / (bins - 1);
                count++;
            }
        }
    }

    void insert(T value) noexcept {
        realMin   = min(realMin, value);
        realMax   = max(realMax, value);
        auto tau  = (value - binMin) / (binMax - binMin);
        tau       = clamp(tau, 0, 1);
        auto idx  = static_cast<uint64_t>(round(tau * (binCount - 1)));
        binY[idx]++;
        elementCount++;
    }

private:
    const T                 binMin;
    const T                 binMax;
    T                       realMin{std::numeric_limits<T>::infinity};
    T                       realMax{-std::numeric_limits<T>::infinity};
    const uint64_t          binCount;
    uint64_t                elementCount{0};
    std::array<T, N>        binX;
    std::array<uint64_t, N> binY;
};


#endif //HISTOGRAM_H
