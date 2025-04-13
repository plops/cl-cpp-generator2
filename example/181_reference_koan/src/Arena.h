#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <vector>
#include "Ref.h"
using namespace std;
template <typename T>
class Arena {
public:
    inline Ref<T> acquire() {
        auto it{find(used.begin(), used.end(), false)};
        if (used.end() == it) { throw runtime_error("no free arena element"); }
        else {
            *it = true;
            auto idx{it - used.begin()};
            auto el{r[idx]};

            return el;
        }
    }
    inline void setUnused(int idx) { used[idx] = false; }
    int         capacity() { return r.size(); }
    int         nb_unused() { return capacity() - nb_used(); }
    int         nb_used() {
        auto sum{0};
        for (auto&& b : used) {
            if (b) { sum++; }
        }
        return sum;
    }
    inline long int use_count(int idx) {
        auto count{r[idx].use_count()};

        return count;
    }
    explicit Arena(int n = 1) : used{vector<bool>(n)}, r{vector<Ref<T>>()}, a{vector<T>(n)} {
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
    vector<bool>   used{};
    vector<Ref<T>> r;
    vector<T>      a;
};
