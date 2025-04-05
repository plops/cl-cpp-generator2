#include <memory>
#include "IWidget.h"
#include "SharedWidget.h"
#include <iostream>
using namespace std;

int main(int argc, char* argv[]) {
    unique_ptr<IWidget> w{make_unique<SharedWidget>(3)};
    cout << w->add(1,2) << endl;
    return 0;
}
