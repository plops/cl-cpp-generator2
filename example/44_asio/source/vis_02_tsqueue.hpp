#ifndef VIS_02_TSQUEUE_H
#define VIS_02_TSQUEUE_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <thread>
#include <deque>
#include <mutex>
;
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
;
// header
template<typename T> class tsqueue  {
        public:
        tsqueue() = default;
        tsqueue(const tsqueue<T>&) = delete;
         ~tsqueue ()    {
                clear();
}
        const T& front ()    {
                        auto lock  = std::scoped_lock(mux_deq);
        return deq.front();
}
        const T& back ()    {
                        auto lock  = std::scoped_lock(mux_deq);
        return deq.back();
}
        bool empty ()    {
                        auto lock  = std::scoped_lock(mux_deq);
        return deq.empty();
}
        size_t size ()    {
                        auto lock  = std::scoped_lock(mux_deq);
        return deq.size();
}
        void clear ()    {
                        auto lock  = std::scoped_lock(mux_deq);
        return deq.clear();
}
        T pop_front ()    {
                        auto lock  = std::scoped_lock(mux_deq);
                        auto el  = std::move(deq.front());
        deq.pop_front();
        return el;
}
        T pop_back ()    {
                        auto lock  = std::scoped_lock(mux_deq);
                        auto el  = std::move(deq.back());
        deq.pop_back();
        return el;
}
        T push_back (const T& item)    {
                        auto lock  = std::scoped_lock(mux_deq);
                        deq.emplace_back(std::move(item));
        std::();
}
        protected:
        std::mutex mux_deq;
        std::deque<T> deq;
};;
#endif