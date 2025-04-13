#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
using namespace std;
template <typename T>
class Arena;
template <typename T>
class Ref {
public:
    explicit Ref(T& r, int idx, Arena<T>& associatedArena) :
        arena{associatedArena}, ref{r}, sp{make_shared<Priv>(idx)} {}
    ~Ref() {
        if (3 == use_count()) { arena.setUnused(idx()); }
    }
    Ref(const Ref& rhs) : arena{rhs.arena}, ref{rhs.ref}, sp{rhs.sp.load()} {}
    Ref(T&&)                            = delete;
    const T&        operator=(const T&) = delete;
    T&              operator=(T&&)      = delete;
    inline long int use_count() { return sp.load().use_count(); }
    inline long int idx() { return sp.load()->idx; }

private:
    class Priv {
    public:
        int idx;
    };
    Arena<T>&                arena;
    T&                       ref;
    atomic<shared_ptr<Priv>> sp{nullptr};
};
template <typename T>
class Arena {
public:
    inline Ref<T> aquire() {
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
    inline void setUnused(int idx) {
        std::cout << "Arena::setUnused" << " idx='" << idx << "' " << std::endl;
        used[idx] = false;
    }
    inline long int use_count(int idx) {
        auto count{r[idx].use_count()};
        std::cout << "Arena::use_count" << " count='" << count << "' " << std::endl;
        return count;
    }
    explicit Arena(int n = 0) : used{vector<bool>(n)}, r{vector<Ref<T>>()}, a{vector<T>(n)} {
        int idx = 0;
        for (auto&& e : a) {
            r.emplace_back(e, idx, *this);
            idx++;
        }
    }
    Arena(const T&)                    = delete;
    Arena(T&&)                         = delete;
    const T&       operator=(const T&) = delete;
    T&             operator=(T&&)      = delete;
    vector<bool>   used{};
    vector<Ref<T>> r;
    vector<T>      a;
};

int main(int argc, char** argv) {
    class Widget {
    public:
    private:
        int   i{3};
        float f{4.5F};
        char  name[20];
    };
    const int n = 3;
    auto      a{Arena<Widget>(n)};
    auto      v{vector<Ref<Widget>>()};
    std::cout << "" << " sizeof(Widget)='" << sizeof(Widget) << "' " << std::endl;
    std::cout << "" << " sizeof(a)='" << sizeof(a) << "' " << std::endl;
    std::cout << "" << " sizeof(a.used)='" << sizeof(a.used) << "' " << std::endl;
    std::cout << "" << " sizeof(a.r)='" << sizeof(a.r) << "' " << std::endl;
    std::cout << "" << " sizeof(a.a)='" << sizeof(a.a) << "' " << std::endl;
    std::cout << "" << " sizeof(v[0])='" << sizeof(v[0]) << "' " << std::endl;
    for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
        auto e{a.aquire()};
        assert(i == e.idx());
        v.push_back(e);
    }
    std::cout << "#### CLEAR ####" << std::endl;
    v.clear();
    std::cout << "#### REUSE N ELEMENTS ####" << std::endl;
    for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
        auto e{a.aquire()};
        assert(i == e.idx());
        v.push_back(e);
    }
    std::cout << "#### TRY TO GET ONE ELEMENT TOO MANY ####" << std::endl;
    v.push_back(a.aquire());
    return 0;
}
