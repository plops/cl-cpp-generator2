
#include <Array.h>
#include <iostream>

#include "VArray.h"
// #include "GeneralArray.h"
using namespace std;
int main(int argc, char *argv[]) {
    Array<int> a(5);
    VArray<int> v(5);
    // GeneralArray<int,Array<int>,ArrayIterator<int>> qaq;
    std::vector<int> q(3);
    auto vv = q.begin();
    int count = 0;
    for (auto &i : a) {
        count ++;
        i=count;
        cout << i << endl;
    }
    for (auto &i : a) {
        cout << i << endl;
    }
    return 0;
}
