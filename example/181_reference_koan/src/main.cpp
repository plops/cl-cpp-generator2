#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
using namespace std;
template <typename T>
class Ref {
public:
    T& get() { return ref; }
    explicit Ref(T& r) : ref{r}, sp{make_shared<Priv>()}, q{new Q} {
        std::cout << "Ref::ctor" << " &ref='" << &ref << "' " << " use_count()='" << use_count() << "' " << std::endl;
    }
    ~Ref() {
        if (q) {
            delete (q);
            q = nullptr;
        }
        std::cout << "~Ref" << " &ref='" << &ref << "' " << " use_count()='" << use_count() << "' " << std::endl;
    }
    Ref(const Ref& rhs) : ref{rhs.ref}, sp{make_shared<Priv>()} { std::cout << "Ref::copy-ctor" << std::endl; }
    int use_count() { return sp.load().use_count(); }

private:
    class Priv {};
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
        if (used.end() == it) { std::cout << "no free arena element" << std::endl; }
        else {
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
    auto          as{array<Widget, N>()};
    auto          ar{deque<Ref<Widget>>()};
    for (auto&& e : as) { ar.emplace_back(e); }
    std::cout << "" << " sizeof(as)='" << sizeof(as) << "' " << std::endl;
    std::cout << "" << " sizeof(ar)='" << sizeof(ar) << "' " << std::endl;
    auto e{ar[0].get()};
    auto qq{ar[0]};
    std::cout << "" << " ar[0].use_count()='" << ar[0].use_count() << "' " << std::endl;
    return 0;
}
