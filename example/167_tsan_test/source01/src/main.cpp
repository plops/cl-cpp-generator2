#include <iostream>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <atomic>

using namespace std;

int main(int argc, char**argv)
{
    //bool keepRunning{true};
    std::atomic<bool> keepRunning{true};
    jthread work{[&]()
    {
        while(keepRunning){};
        cout << "end worker" <<endl;
    }};
    jthread q{[&]()
    {
        cout << "stop worker" << endl;
        keepRunning = false;
    }};
    q.join();
    work.join();
    return 0;
}