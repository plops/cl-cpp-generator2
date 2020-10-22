#ifndef VIS_01_MESSAGE_H
#define VIS_01_MESSAGE_H
#include "utils.h"
;
#include "globals.h"
;
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
;
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
;
#include "vis_03_connection.hpp"
;
// header
#include "vis_03_connection.hpp"
;
template <typename T> class message_header {
public:
  T id{};
  uint32_t size = 0;
};
#endif