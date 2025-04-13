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
    explicit Ref(T& r, int index, Arena<T>& associatedArena) : arena{associatedArena}, ref{r}, idx{index} {
        std::cout << "Ref-ctor" << " &arena='" << &arena << "' " << " &ref='" << &ref << "' " << " idx='" << idx << "' "
                  << std::endl;
    }
    int getIndex() const { return idx; }
    ~Ref() { std::cout << "Ref-dtor" << " idx='" << idx << "' " << std::endl; }
    Ref(const T&)                = delete;
    Ref(T&&)                     = delete;
    const T& operator=(const T&) = delete;
    T&       operator=(T&&)      = delete;
C
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
    auto      v{vector<Widget>(3)};
    auto      e0{make_shared<Ref<Widget>>(v[0], 0, a)};
    auto      e1{make_shared<Ref<Widget>>(v[1], 1, a)};
    auto      e2{make_shared<Ref<Widget>>(v[2], 2, a)};
    e1 = e0;
    auto c0{e0};
    auto c1{move(e1)};
    return 0;
}
