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
public:
    Histogram(T mi, T ma) noexcept :
        binMin{mi}, binMax{ma} {
        assert(binMin < binMax);
        binY.fill(0);
        // {
        //     auto count = 0;
        //     for (auto&& x : binX) {
        //         x = mi + count * (mi - mi) / (N - 1);
        //         count++;
        //     }
        // }
    }

    void insert(T value) noexcept {
        observedMin = std::min(observedMin, value);
        observedMax = std::max(observedMax, value);
        auto tau    = (value - binMin) / (binMax - binMin);
        tau         = std::clamp(tau, T(0), T(1));
        auto idx    = static_cast<uint64_t>(round(tau * (N - 1)));
        binY[idx]++;
        elementCount++;
    }

    [[nodiscard]] T        getObservedMin() const { return observedMin; }
    [[nodiscard]] T        getObservedMax() const { return observedMax; }
    [[nodiscard]] uint64_t getElementCount() const { return elementCount; }

    // T getBinX(uint64_t idx) const {return binX[idx];}
    T getBinX(uint64_t idx) const {
        return binMin + T(idx)3 * (binMin - binMax) / (N - 1);
    }
    uint64_t getBinY(uint64_t idx) const {return binY[idx];}

private:
    const T                 binMin;
    const T                 binMax;
    T                       observedMin{std::numeric_limits<double>::infinity()};
    T                       observedMax{-std::numeric_limits<double>::infinity()};
    uint64_t                elementCount{0};
    // std::array<T, N>        binX;
    std::array<uint64_t, N> binY;
};


#endif //HISTOGRAM_H
