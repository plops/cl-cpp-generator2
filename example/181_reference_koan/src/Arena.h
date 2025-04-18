#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>
#include "Ref.h"
using namespace std;
template <typename T>
class Arena {
public:
    int firstUnused() {
        auto l{lock_guard(m)};
        auto it{find(used.begin(), used.end(), false)};
        if (used.end() == it) { return -1; }
        return it - used.begin();
    }
    inline Ref<T> acquire() {
        elementNowUnused.clear();
        auto idx{firstUnused()};
        if (-1 == idx) {
            // pikus p.549
            std::cout << "waiting for element to become unused" << std::endl;
            elementNowUnused.wait(false, memory_order_acquire);
            // according to standard this wait should not spuriously wake up. the book still adds this check because
            // tsan thinks otherwise new elements should now be present
            auto idx{firstUnused()};
            if (-1 == idx) { throw runtime_error("no free arena element"); }
            auto el{r.at(idx)};
            std::cout << "found unused element after wait" << " idx='" << idx << "' " << std::endl;
            {
                auto l{lock_guard(m)};
                used[idx] = true;
            }
            return el;
        }
        auto el{r.at(idx)};
        std::cout << "found unused element" << " idx='" << idx << "' " << std::endl;
        {
            auto l{lock_guard(m)};
            used[idx] = true;
        }
        return el;
    }
    inline void setUnused(int idx) {
        {
            auto l{lock_guard(m)};
            std::cout << "Arena::setUnused" << " idx='" << idx << "' " << std::endl;
            used[idx] = false;
        }
        elementNowUnused.test_and_set(memory_order_release);
        elementNowUnused.notify_one();
    }
    int capacity() const { return r.size(); }
    int nb_unused() const { return capacity() - nb_used(); }
    int nb_used() {
        auto sum{0};
        for (auto&& b : used) {
            if (b) { sum++; }
        }
        return sum;
    }
    explicit Arena(int n = 1) : r{vector<Ref<T>>()}, a{vector<T>(n)} {
        {
            auto l{lock_guard(m)};
            used.reserve(n);
            used.resize(n);
            ranges::fill(used, false);
        }
        int idx = 0;
        for (auto&& e : a) {
            r.emplace_back(e, idx, *this);
            idx++;
        }
    }
    Arena(const T&)              = delete;
    Arena(T&&)                   = delete;
    const T& operator=(const T&) = delete;
    T&       operator=(T&&)      = delete;

private:
    mutex          m; // protect access to used[] and idx in Ref<T>
    vector<bool>   used;
    vector<Ref<T>> r;
    vector<T>      a;
    atomic_flag    elementNowUnused{false};
};
