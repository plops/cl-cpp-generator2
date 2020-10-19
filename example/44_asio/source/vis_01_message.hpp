#ifndef VIS_01_MESSAGE_H
#define VIS_01_MESSAGE_H
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
// header;
template<<typename T>> class message_header  {
        T id{};
        uint32_t size = 0;
};
#endif