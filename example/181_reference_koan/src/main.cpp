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
    explicit Ref(T& r) : ref{r}, sp{make_shared<Priv>()}, q{new Q} { std::cout << "Ref::ctor" << std::endl; }
    ~Ref() {
        delete (q);
        std::cout << "~Ref" << std::endl;
    }
    Ref(const T& rhs) : ref{rhs.ref}, sp{make_shared<Priv>()} { std::cout << "Ref::copy-ctor" << std::endl; }
    int use_count() { return sp.load().use_count(); }

private:
    class Priv {};
    class Q {
    private:
        mutex              m;
        condition_variable c;
    };
    T&                       ref;
    atomic<shared_ptr<Priv>> sp;
    Q*                       q;
};

int main(int argc, char** argv) {
    class Widget {
    public:
    private:
        int   i{3};
        float f{4.5F};
    };
    constexpr int N{17};
    auto          as{array<Widget, N>()};
    auto          ar{deque<Ref<Widget>>()};
    for (auto&& e : as) { ar.emplace_back(e); }
    std::cout << "" << " sizeof(as)='" << sizeof(as) << "' " << std::endl;
    std::cout << "" << " sizeof(ar)='" << sizeof(ar) << "' " << std::endl;
    std::cout << "" << " ar[0].use_count()='" << ar[0].use_count() << "' " << std::endl;
    return 0;
}
