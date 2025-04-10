#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
using namespace std;
template <typename T>
class Ref {
public:
    explicit Ref(T& r) : ref{r}, sp{createPriv()}, q{new Q} {
        std::cout << "Ref::ctor" << " sp.load().get()='" << sp.load().get() << "' " << " &ref='" << &ref << "' "
                  << " use_count()='" << use_count() << "' " << std::endl;
    }
    ~Ref() {
        if (q) {
            delete (q);
            q = nullptr;
        }
        std::cout << "~Ref" << " &ref='" << &ref << "' " << " use_count()='" << use_count() << "' " << std::endl;
    }
    Ref(const Ref& rhs) : ref{rhs.ref}, sp{createPriv()} { std::cout << "Ref::copy-ctor" << std::endl; }
    int use_count() { return sp.load().use_count(); }

private:
    class Priv {
    public:
        int idx;
    };
    shared_ptr<Priv> createPriv() {
        return shared_ptr<Priv>(new Priv(3), [&](Priv* p) {
            std::cout << "~shared_ptr" << " p='" << p << "' " << " p->idx='" << p->idx << "' " << std::endl;
            delete (p);
        });
    }
    class Q {
    private:
        mutex              m;
        condition_variable c;
    };
    T&                       ref;
    atomic<shared_ptr<Priv>> sp{nullptr};
    Q*                       q{nullptr};
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
            return el;
        }
    }
    Arena() {
        for (auto&& e : a) { r.emplace_back(e); }
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
    constexpr int N{3};
    auto          a{Arena<Widget, N>()};
    auto          v{deque<Ref<Widget>>()};
    for (decltype(0 + N + 1 + 1) i = 0; i < N + 1; i += 1) { v.push_back(a.aquire()); }
    return 0;
}
