#include <array>
#include <iostream>
using namespace std;
template <typename T>
class Ref {
public:
    explicit Ref(T& r) : ref{r} {}
    ~Ref() {}
    Ref(const T& rhs) : ref{rhs.ref} {}

private:
    T& ref;
};

int main(int argc, char** argv) {
    class Widget {
    public:
    private:
        int   i{3};
        float f{4.5F};
    };
    constexpr int N{17};
    auto          a{array<Widget, N>()};
    auto          ar{array<Ref<Widget>, N>()};
    return 0;
}
