#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
using namespace std;
constexpr int N{3};
template <typename T, int N>
class Arena;
template <typename T>
class Ref {
public:
    explicit Ref(T& r, int idx, Arena<T, N>& arena) : ref{r}, sp{createPriv(idx, arena)} {
        std::cout << "Ref::ctor" << " idx='" << idx << "' " << " sp.load().get()='" << sp.load().get() << "' "
                  << " &ref='" << &ref << "' " << " &arena='" << &arena << "' " << std::endl;
    }
    ~Ref() {
        if (1 == sp.load().use_count()) {
            std::cout << "#### # tell arena" << " sp.load()->idx='" << sp.load()->idx << "' "
                      << " sp.load().use_count()='" << sp.load().use_count() << "' " << std::endl;
            sp.load()->arena.setUnused(sp.load()->idx);
        }
    }
    Ref(const Ref& rhs) : ref{rhs.ref}, sp{createPriv(rhs.sp.load()->idx, rhs.sp.load()->arena)} {
        std::cout << "Ref::copy-ctor" << " sp.load()->idx='" << sp.load()->idx << "' " << std::endl;
    }
    long int use_count() { return sp.load().use_count(); }

private:
    class Priv {
    public:
        int          idx;
        Arena<T, N>& arena;
    };
    shared_ptr<Priv> createPriv(int idx, Arena<T, N>& arena) {
        return shared_ptr<Priv>(new Priv(idx, arena), [&](Priv* p) {
            std::cout << "~shared_ptr" << " p='" << p << "' " << " p->idx='" << p->idx << "' " << std::endl;
            delete (p);
        });
    }
    T&                       ref;
    atomic<shared_ptr<Priv>> sp{nullptr};
};
template <typename T, int N>
class Arena {
public:
    Ref<T> aquire() {
        auto it{find(used.begin(), used.end(), false)};
        if (used.end() == it) { throw runtime_error("no free arena element"); }
        else {
            *it = true;
            auto idx{it - used.begin()};
            auto el{r.at(idx)};
            std::cout << "found unused element" << " idx='" << idx << "' " << std::endl;
            return el;
        }
    }
    void     setUnused(int idx) { used[idx] = false; }
    long int use_count(int idx) { return r[idx].use_count(); }
    Arena() {
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
    array<T, N>    a;
    array<bool, N> used{};
    deque<Ref<T>>  r;
};

int main(int argc, char** argv) {
    class Widget {
    public:
    private:
        int   i{3};
        float f{4.5F};
    };
    auto a{Arena<Widget, N>()};
    auto v{deque<Ref<Widget>>()};
    for (decltype(0 + N + 1) i = 0; i < N; i += 1) { v.push_back(a.aquire()); }
    std::cout << "#### CLEAR ####" << std::endl;
    v.clear();
    for (decltype(0 + N + 1 + 1) i = 0; i < N + 1; i += 1) { v.push_back(a.aquire()); }
    return 0;
}
