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
    void setUnused(int idx) {}
};

int main(int argc, char** argv) {
    class Widget {
    public:
    private:
        int   i{3};
        float f{4.5F};
    };
    const int n = 3;
    auto      a{Arena<Widget>()};
    auto      v{vector<Widget>(1)};
    auto      e{make_shared<Ref<Widget>>(v[0], 0, a)};
    return 0;
}
