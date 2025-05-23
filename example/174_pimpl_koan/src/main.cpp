#include <memory>
#include <iostream>
#include "IWidget.h"
#include "SharedWidget.h"
#include "UniqueWidget.h"
#include "IContainer.h"
#include "Vec.h"

using namespace std;

int main(int argc, char* argv[]) {
    unique_ptr<IContainer> vv= make_unique<Vec>(3);
    unique_ptr<IContainer> vv2 =make_unique<Vec>(4);
    unique_ptr<IContainer> vv3 =make_unique<Vec>(5);
    unique_ptr<IContainer> vv4= make_unique<Vec>(6);

    unique_ptr<IWidget> w{make_unique<SharedWidget>(3,move(vv))};
    unique_ptr<IWidget> w2{make_unique<UniqueWidget>(3,move(vv2))};

    cout << w->add(1,2) << endl;
    w->insert(4);
    cout << w2->add(1,2) << endl;
    array<unique_ptr<IWidget>,2> a={make_unique<SharedWidget>(4,move(vv3)), make_unique<UniqueWidget>(5,move(vv4))};
    for (auto&& e : a)
        cout << e->add(1,2) << endl;
    return 0;
}
