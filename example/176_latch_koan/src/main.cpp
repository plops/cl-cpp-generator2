#include <iostream>
#include <latch>
#include <thread>
#include <string>
#include <string_view>
#include <memory>
using namespace std;

class Worker
{
public:
    Worker(std::string_view n, shared_ptr<latch> workDone_, shared_ptr<latch> goHome_)
        : name(n), workDone{move(workDone_)}, goHome{move(goHome_)}
    {
    }

    void operator()()
    {
        workDone->count_down(); // notify boss when work is done
        goHome->wait(); // wait before going home
    }

private:
    string name;
    shared_ptr<latch> workDone;
    shared_ptr<latch> goHome;
};

int main(int argc, char* argv[])
{
    auto workDone{make_shared<latch>(3)};
    auto goHome{make_shared<latch>(1)};

    cout << "Start working" << endl;

    Worker herb("Herb", workDone, goHome);
    thread herbWork(herb);

    Worker scott("Scott", workDone, goHome);
    thread scottWork(scott);

    Worker bjarne("Bjarne", workDone, goHome);
    thread bjarneWork(bjarne);

    workDone->wait();

    cout << "workDone" << endl;

    goHome->count_down();

    cout << "goHome" << endl;

    herbWork.join();
    scottWork.join();
    bjarneWork.join();

    cout << "finished" << endl;
    return 0;
}
