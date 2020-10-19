
#include "utils.h"

#include "globals.h"

extern State state;
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
#include <chrono>
#include <iostream>
#include <thread>
using namespace std::chrono_literals;

// implementation
std::vector<char> buffer(20 * 1024);
void grab_some_data(boost::asio::ip::tcp::socket &socket) {
  socket.async_read_some(boost::asio::buffer(buffer.data(), buffer.size()),
                         [&](std::error_code ec, std::size_t length) {
                           if (!(ec)) {

                             (std::cout)
                                 << (std::setw(10))
                                 << (std::chrono::high_resolution_clock::now()
                                         .time_since_epoch()
                                         .count())
                                 << (" ") << (std::this_thread::get_id())
                                 << (" ") << (__FILE__) << (":") << (__LINE__)
                                 << (" ") << (__func__) << (" ")
                                 << ("read bytes:") << (" ") << (std::setw(8))
                                 << (" length='") << (length) << ("'")
                                 << (std::endl) << (std::flush);
                             for (size_t i = 0; (i) < (length); (i)++) {
                               (std::cout) << (buffer[i]);
                             }
                             // will wait until some data available
                             grab_some_data(socket);
                           }
                         });
}
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  boost::system::error_code ec;
  boost::asio::io_context context;
  auto idle_work = boost::asio::io_context::work(context);
  auto context_thread = std::thread([&]() { context.run(); });
  auto endpoint = boost::asio::ip::tcp::endpoint(
      boost::asio::ip::make_address("93.184.216.34", ec), 80);
  auto socket = boost::asio::ip::tcp::socket(context);
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
    grab_some_data(socket);
    auto request = std::string("GET /index.html HTTP/1.1\r\nHost: "
                               "example.com\r\nConnection: close\r\n\r\n");
    socket.write_some(boost::asio::buffer(request.data(), request.size()), ec);
    std::this_thread::sleep_for(2000ms);
  }
  return 0;
}