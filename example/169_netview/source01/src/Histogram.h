//
// Created by martin on 3/16/25.
//

#ifndef HISTOGRAM_H
#define HISTOGRAM_H
#include <array>
#include <limits>
#include <ostream>

/**
 * @brief Compute histogram, maximum and minimum of data
 *
 * @note Initialize the histogram range using the constructor arguments binMin and binMax.
 *       The histogram is updated using insert(T). Elements that are out of range are counted into the closest bin.
 *       You can check observed{Max,Min} to see if the observed data fell outside the range of histogram bins.
 *
 * @tparam T Datatype to be fed into the histogram (typically double)
 * @tparam N Number of bins of the histogram (e.g. 128)
 */
template <typename T, int N>
class Histogram {
public:
    Histogram(T mi, T ma) noexcept :
        binMin{mi}, binMax{ma}, scale{1.0 / (binMax - binMin)} {
        assert(binMin < binMax);
        binY.fill(0);
    }

    void insert(T value) noexcept {
        observedMin = std::min(observedMin, value);
        observedMax = std::max(observedMax, value);
        auto tau    = (value - binMin) * scale;
        tau         = std::clamp(tau, .0, 1.);
        auto idx    = static_cast<uint64_t>(round(tau * (N - 1)));
        ++binY[idx];
        elementCount++;
    }

    [[nodiscard]] T        getObservedMin() const { return observedMin; }
    [[nodiscard]] T        getObservedMax() const { return observedMax; }
    [[nodiscard]] T        getBinMin() const { return binMin; }
    [[nodiscard]] T        getBinMax() const { return binMax; }
    [[nodiscard]] uint64_t getElementCount() const { return elementCount; }

    T        getBinX(uint64_t idx) const { return binMin + T(idx) * (binMax - binMin) / (N - 1); }
    uint64_t getBinY(uint64_t idx) const { return binY[idx]; }

    friend std::ostream& operator<<(std::ostream& Os, const Histogram& Obj) {
        Os << "observedMin: " << Obj.observedMin << '\n'
                << "observedMax: " << Obj.observedMax << '\n'
                << "elementCount: " << Obj.elementCount << '\n';
        auto s = 1.0 / static_cast<double>(Obj.elementCount);
        for (auto i = 0; i < N; ++i) {
            auto density = static_cast<double>(Obj.getBinY(i)) * s * 99;
            Os << '\t' << Obj.getBinX(i) << '\t' << density << '\n';
        }
        return Os;
    }

private:
    const T                 binMin;
    const T                 binMax;
    const double            scale;
    T                       observedMin{std::numeric_limits<T>::max()};
    T                       observedMax{-std::numeric_limits<T>::max()};
    uint64_t                elementCount{0};
    std::array<uint64_t, N> binY;
};
#endif //HISTOGRAM_H
