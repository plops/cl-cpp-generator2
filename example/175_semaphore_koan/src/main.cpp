#include <iostream>
#include <semaphore>
#include <thread>
#include <vector>

using namespace std;

int main(int argc, char* argv[]) {
    vector<int> vec{};
    counting_semaphore<1> prepareSignal(0);

    auto prepareWork = [&]()
    {
        vec.insert(vec.end(), {0,1,0,3});
        cout << "Sender: Data prepared." << endl;
        prepareSignal.release();
    };

    auto completeWork = [&]()
    {
        cout << "Waiter: Waiting for data." << endl;
        prepareSignal.acquire();
        vec[2] = 2;
        cout << "Waiter: Complete the work." << endl;
        for (auto e: vec)
            cout << e << " ";
        cout << endl;
    };

    cout << endl;

    thread t1(prepareWork);
    thread t2(completeWork);

    t1.join();
    t2.join();

    cout << endl;

    return 0;
}
