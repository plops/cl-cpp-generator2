#include <chrono>
#include <gtest/gtest.h>
#include <latch>
#include <thread>
#include <vector>
#include "Arena.h"
using namespace std;
using namespace std::chrono_literals;
TEST(Arena, acquire_perform_freeElementsShrink) {
    struct Widget {
        int   i;
        float a;
    };
    auto n{3};
    auto a{Arena<Widget>(n)};
    auto v{vector<Ref<Widget>>()};
    for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
        v.push_back(a.acquire());
        EXPECT_EQ(a.capacity(), n);
        EXPECT_EQ(a.nb_used(), 1 + i);
    }
};
TEST(Arena, acquire_performUntilWait_elementArrivesAfterWait) {
    struct Widget {
        int   i;
        float a;
    };
    auto n{3};
    auto a{Arena<Widget>(n)};
    auto la{latch(1)};
    auto th{jthread([&n, &a, &la]() {
        auto v{vector<Ref<Widget>>()};
        for (decltype(0 + n + 1) i = 0; i < n; i += 1) {
            v.push_back(a.acquire());
            EXPECT_EQ(a.capacity(), n);
            EXPECT_EQ(a.nb_used(), 1 + i);
        }
        la.count_down();
        this_thread::sleep_for(10ms);
        std::cout << "exiting thread that held elements" << std::endl;
    })};
    la.wait();
    // wait until the thread used all the elements
    auto start{chrono::system_clock::now()};
    a.acquire();
    auto end{chrono::system_clock::now()};
    auto duration_ms{(end - start).count() * 1.00e-6F};
    std::cout << "" << " duration_ms='" << duration_ms << "' " << std::endl;
    EXPECT_GE(duration_ms, 10);
    EXPECT_LE(duration_ms, 12);
};
