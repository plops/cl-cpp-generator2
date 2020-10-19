
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
};