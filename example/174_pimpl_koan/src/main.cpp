#include <memory>
// #include "IWidget.h"
#include "Widget.h"
#include <iostream>
using namespace std;

int main(int argc, char* argv[]) {
    // unique_ptr<IWidget>
    auto w{make_unique<Widget>(3)};
    cout << w->add(1,2) << endl;
    return 0;
}
