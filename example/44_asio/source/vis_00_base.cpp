
#include "utils.h"

#include "globals.h"

extern State state;
#include <asio.hpp>
#include <asio/ts/buffer.hpp>
#include <asio/ts/internet.hpp>
#include <chrono>
#include <iostream>
#include <thread>

// implementation
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  asio::error_code ec;
  asio::io_context context;
  auto endpoint =
      asio::ip::tcp::endpoint(asio::ip::make_address("93.184.216.34", ec), 80);
  auto socket = asio::ip::tcp::socket(context);
  socket.connect(endpoint, ec);
  if (ec) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("failed to connect to address") << (" ") << (std::setw(8))
                << (" ec.message()='") << (ec.message()) << ("'") << (std::endl)
                << (std::flush);
  } else {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("connected") << (" ") << (std::endl) << (std::flush);
  }
  if (socket.is_open()) {
    auto request = std::string("GET /index.html HTTP/1.1\r\nHost: "
                               "example.com\r\nConnection: close\r\n\r\n");
    socket.write_some(asio::buffer(request.data(), request.size()), ec);
    auto bytes = socket.available();

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("bytes available") << (" ") << (std::setw(8))
                << (" bytes='") << (bytes) << ("'") << (std::endl)
                << (std::flush);
  }
  return 0;
}