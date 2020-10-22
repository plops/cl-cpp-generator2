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
*682 * template <typename T> class message_header {
public:
  T id{};
  uint32_t size = 0;
};
*682 * template <typename T> class message {
public:
  message_header<T> header{};
  std::vector<uint8_t> body;
  *332 * size_t size() const;
  *332 * template <typename DataType>
         friend message<T> &operator<<(message<T> &msg, const DataType &data);
  *332 * template <typename DataType>
         friend message<T> &operator>>(message<T> &msg, DataType &data);
};
*682 * template <typename T> class owned_message {
public:
  std::shared_ptr<connection<T>> remote = nullptr;
  message<T> msg;
};
// header

template <typename T> class connection;
;
#endif