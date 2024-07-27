#include <readerwritercircularbuffer.h>
#include <cassert>
#include <iostream>
#include <random>
#include <thread>


int main()
{
    moodycamel::BlockingReaderWriterCircularBuffer<int> q(1024);  // pass initial capacity

    q.try_enqueue(1);
    int number;
    q.try_dequeue(number);
    assert(number == 1);

    q.wait_enqueue(123);
    q.wait_dequeue(number);
    assert(number == 123);

    q.wait_dequeue_timed(number, std::chrono::milliseconds(10));

    // create two jthread, one produces random numbers with random delays from 1 to 20ms and the other consumes them

    std::jthread producer([&q] {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 20);
        for (int i = 0; i < 100; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen)));
            q.wait_enqueue(i);
        }
    });

    std::jthread consumer([&q] {
        int number;
        for (int i = 0; i < 100; ++i) {
            q.wait_dequeue(number);
            std::cout << number << std::endl;
        }
    });

    return 0;
}
