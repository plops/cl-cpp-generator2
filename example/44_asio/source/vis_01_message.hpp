#ifndef VIS_01_MESSAGE_H
#define VIS_01_MESSAGE_H
*1079 * #include "utils.h";
*1079 * #include "globals.h";
*1079 * #include<iostream>
#include <chrono>
#include <thread>
#include <vector>
    ;
*1079 * #include<boost / asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
    ;
*1079 * *682 * template <typename T> class message_header {
public:
  T id{};
  uint32_t size = 0;
};
*1079 * *682 * template <typename T> class message {
public:
  message_header<T> header{};
  std::vector<uint8_t> body;
  *332 * size_t size() const;
  *332 * template <typename DataType>
         friend message<T> &operator<<(message<T> &msg, const DataType &data);
  *332 * template <typename DataType>
         friend message<T> &operator>>(message<T> &msg, DataType &data);
};
*1079 * *682 * template <typename T> class owned_message {
public:
  std::shared_ptr<connection<T>> remote = nullptr;
  message<T> msg;
};
*1079 * // header

    template <typename T>
    class connection;
;
#endif