#include <memory>
#include "IWidget.h"
#include "SharedWidget.h"
#include "UniqueWidget.h"
#include <iostream>

#include <array>
using namespace std;

int main(int argc, char* argv[]) {
    unique_ptr<IWidget> w{make_unique<SharedWidget>(3)};
    unique_ptr<IWidget> w2{make_unique<UniqueWidget>(3)};

    cout << w->add(1,2) << endl;
    cout << w2->add(1,2) << endl;

    array<unique_ptr<IWidget>,2> a={make_unique<SharedWidget>(4), make_unique<UniqueWidget>(5)};

    for (auto&& e : a)
        cout << e->add(1,2) << endl;
    return 0;
}
