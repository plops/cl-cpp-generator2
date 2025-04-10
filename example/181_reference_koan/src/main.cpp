#include <array>
#include <atomic>
#include <cassert>
#include <deque>
#include <iostream>
#include <memory>
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
        if (2 == sp.load().use_count()) {
            std::cout << "#### # tell arena" << " idx()='" << idx() << "' " << " use_count()='" << use_count() << "' "
                      << std::endl;
            sp.load()->arena.setUnused(idx());
        }
    }
    Ref(const Ref& rhs) : ref{rhs.ref}, sp{rhs.sp.load()} {
        std::cout << "Ref::copy-ctor" << " sp.load()->idx='" << sp.load()->idx << "' " << std::endl;
    }
    long int use_count() { return sp.load().use_count(); }
    long int idx() { return sp.load()->idx; }

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
    void setUnused(int idx) {
        std::cout << "Arena::setUnused" << " idx='" << idx << "' " << std::endl;
        used[idx] = false;
    }
    long int use_count(int idx) {
        auto count{r[idx].use_count()};
        std::cout << "Arena::use_count" << " count='" << count << "' " << std::endl;
        return count;
    }
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
    for (decltype(0 + N + 1) i = 0; i < N; i += 1) {
        auto e{a.aquire()};
        assert(i == e.idx());
        v.push_back(e);
    }
    std::cout << "#### CLEAR ####" << std::endl;
    v.clear();
    std::cout << "#### REUSE N ELEMENTS ####" << std::endl;
    for (decltype(0 + N + 1) i = 0; i < N; i += 1) {
        auto e{a.aquire()};
        assert(i == e.idx());
        v.push_back(e);
    }
    std::cout << "#### TRY TO GET ONE ELEMENT TOO MANY ####" << std::endl;
    v.push_back(a.aquire());
    return 0;
}
