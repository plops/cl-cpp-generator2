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
class Ref : public enable_shared_from_this<Ref<T>> {
public:
    explicit Ref(T& r, int idx, Arena<T>& associatedArena) : ref{r}, arena{associatedArena} {}
    int getIndex() { return idx; }
    ~Ref() {
        auto sp{this->shared_from_this()};
        if (3 == sp.use_count()) { arena.setUnused(idx); }
    }
    Ref(const Ref& rhs) : ref{rhs.ref}, arena{rhs.arena}, idx{rhs.idx} {}
    Ref(T&&)                     = delete;
    const T& operator=(const T&) = delete;
    T&       operator=(T&&)      = delete;

private:
    Arena<T>& arena;
    T&        ref;
    int       idx;
};
template <typename T>
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
    Arena(int n = 0) : a{vector<T>(n)}, used{vector<bool>(n)}, r{vector<Ref<T>>()} {
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
    vector<T>                          a;
    vector<bool>                       used{};
    vector<atomic<shared_ptr<Ref<T>>>> r;
};

int main(int argc, char** argv) {
    class Widget {
    public:
    private:
        int   i{3};
        float f{4.5F};
    };
    const int n = 3;
    auto      a{Arena<Widget>(n)};
    auto      v{vector<Ref<Widget>>()};
    for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
        auto e{a.aquire()};
        assert(i == e.getIndex());
        v.push_back(e);
    }
    std::cout << "#### CLEAR ####" << std::endl;
    v.clear();
    std::cout << "#### REUSE N ELEMENTS ####" << std::endl;
    for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
        auto e{a.aquire()};
        assert(i == e.getIndex());
        v.push_back(e);
    }
    std::cout << "#### TRY TO GET ONE ELEMENT TOO MANY ####" << std::endl;
    v.push_back(a.aquire());
    return 0;
}
