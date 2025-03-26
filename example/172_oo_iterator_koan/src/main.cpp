
#include <iostream>
#include <Array.h>
using namespace std;
int main(int argc, char *argv[]) {
    Array<int> a(5);

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
