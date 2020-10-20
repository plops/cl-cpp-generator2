#ifndef VIS_02_TSQUEUE_H
#define VIS_02_TSQUEUE_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
;
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
;
// header
template<typename T> class tsqueue  {
        public:
        size_t size () const   {
                return sizeof(((sizeof(message_header<T>))+(body.size())));
}
        protected:
        std::mutex mutex;
        std::deque<T> deque;
};;
#endif