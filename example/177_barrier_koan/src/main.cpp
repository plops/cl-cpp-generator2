#include <iostream>
#include <barrier>
#include <thread>
#include <string>
#include <string_view>
// #include <memory>

using namespace std;

barrier workDone(3);

class Worker
{
public:
    Worker(std::string_view n) //, shared_ptr<std::barrier> workDone_)
        : name(n) //, workDone{move(workDone_)}
    {
    }
    virtual void operator()() = 0;

   // shared_ptr<std::barrier> workDone;

private:
    string name;
};

class FullTimeWorker :public Worker
{
public:
    FullTimeWorker(string_view n) : Worker(n) {}
    void operator()() override
    {
        workDone.arrive_and_wait();
        workDone.arrive_and_wait();
    }
};

class PartTimeWorker :public Worker
{
public:
    PartTimeWorker(string_view n) : Worker(n) {}
    void operator()() override
    {
        workDone.arrive_and_drop();
    }
};

int main(int argc, char* argv[])
{

    cout << "Start working" << endl;

    FullTimeWorker herb("Herb"); //, workDone);
    thread herbWork(herb);

    FullTimeWorker scott("Scott"); //, workDone);
    thread scottWork(scott);

    PartTimeWorker bjarne("Bjarne"); // , workDone);
    thread bjarneWork(bjarne);

    cout << "workDone" << endl;

    herbWork.join();
    scottWork.join();
    bjarneWork.join();

    cout << "finished" << endl;
    return 0;
}
