
#include "utils.h"

#include "globals.h"

extern State state;
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

// implementation
template <typename T>;
template <typename T> size_t message::size() const {
  return sizeof(((sizeof(message_header<T>)) + (body.size())));
}
template <typename DataType>
friend message<T> &message::operator<<(message<T> &msg, const DataType &data) {
  static_assert(std::is_standard_layout<DataType>::value,
                "data is too complicated");
  auto i = msg.body.size();
  msg.body.resize(((msg.body.size()) + (sizeof(DataType))));
  std::memcpy(((msg.body.data()) + (i)), &data, sizeof(DataType));
  msg.header.size = msg.size();
  return msg;
};