#pragma once
#include <atomic>
#include <memory>
#include <mutex>
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
    Ref& operator=(const Ref& rhs) {
        if (!(this == &rhs)) {
            arena = rhs.arena;
            ref   = rhs.ref;
            sp    = rhs.sp.load();
        }
        return *this;
    }
    Ref(Ref&& rhs) noexcept : arena{rhs.arena}, ref{rhs.ref}, sp{rhs.sp.load()} {}
    Ref& operator=(Ref&& rhs) noexcept {
        if (!(this == &rhs)) {
            arena = rhs.arena;
            ref   = rhs.ref;
            sp    = rhs.sp.load();
        }
        return *this;
    }
    inline long int use_count() { return sp.load().use_count(); }

private:
    inline long int idx() { return sp.load()->idx; }
    class Priv {
    public:
        int idx;
    };
    Arena<T>&                arena;
    T&                       ref;
    atomic<shared_ptr<Priv>> sp{nullptr};
};
