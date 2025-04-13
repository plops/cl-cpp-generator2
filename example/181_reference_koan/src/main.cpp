#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <vector>
#include "Arena.h"
using namespace std;

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
    std::cout << "" << " sizeof(v[0])='" << sizeof(v[0]) << "' " << std::endl;
    for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
        auto e{a.acquire()};
        assert(i == e.idx());
        v.push_back(e);
    }
    std::cout << "#### CLEAR ####" << std::endl;
    v.clear();
    std::cout << "#### REUSE N ELEMENTS ####" << std::endl;
    for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
        auto e{a.acquire()};
        assert(i == e.idx());
        v.push_back(e);
    }
    std::cout << "#### TRY TO GET ONE ELEMENT TOO MANY ####" << std::endl;
    v.push_back(a.acquire());
    return 0;
}
