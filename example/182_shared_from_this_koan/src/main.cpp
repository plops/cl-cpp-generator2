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
    explicit Ref(T& r, int index, Arena<T>& associatedArena) :
        enable_shared_from_this<Ref<T>>{}, arena{associatedArena}, ref{r}, idx{index} {}
    int getIndex() const { return idx; }
    ~Ref() {
        if (3 == this->shared_from_this().use_count()) { arena.setUnused(idx); }
    }
    Ref(const Ref& rhs) : enable_shared_from_this<Ref<T>>{rhs}, arena{rhs.arena}, ref{rhs.ref}, idx{rhs.idx} {}
    Ref& operator=(const Ref& rhs) {
        if (this == &rhs) { return *this; }
        enable_shared_from_this<Ref<T>>::operator=(rhs);
        arena = rhs.arena;
        ref   = rhs.ref;
        idx   = rhs.idx;
        return *this;
    }
    Ref(Ref&& rhs) noexcept : enable_shared_from_this<Ref<T>>{rhs}, arena{rhs.arena}, ref{rhs.ref}, idx{rhs.idx} {}
    Ref& operator=(Ref&& rhs) noexcept {
        if (this == &rhs) { return *this; }
        enable_shared_from_this<Ref<T>>::operator=(move(rhs));
        arena = rhs.arena;
        ref   = rhs.ref;
        idx   = rhs.idx;
        return *this;
    }

private:
    Arena<T>& arena;
    T&        ref;
    int       idx;
};
template <typename T>
class Arena {
public:
    using SRef = atomic<shared_ptr<Ref<T>>>;
    void     setUnused(int idx) { used[idx] = false; }
    long int use_count(int idx) {
        auto count{r.at(idx).use_count()};
        return count;
    }
    Arena(int n = 0) : a{vector<T>(n)}, used{vector<bool>(n)}, r{vector<SRef>()} {
        int idx = 0;
        for (auto&& e : a) {
            r.push_back(make_shared<Ref<T>>(e, idx, *this));
            idx++;
        }
    }
    Arena(const T&)              = delete;
    Arena(T&&)                   = delete;
    const T& operator=(const T&) = delete;
    T&       operator=(T&&)      = delete;

private:
    vector<T>    a;
    vector<bool> used{};
    vector<SRef> r;
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
    return 0;
}
