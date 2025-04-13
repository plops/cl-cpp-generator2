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


    for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
        auto e{a.aquire()};
        assert(i == e.idx());
        v.push_back(e);
    }

    v.clear();

    for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
        auto e{a.aquire()};
        assert(i == e.idx());
        v.push_back(e);
    }

    v.push_back(a.aquire());
    return 0;
}
