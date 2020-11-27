
#include "utils.h"

#include "globals.h"

extern State state;
#include "vis_01_message.hpp"
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
#include <chrono>
#include <iostream>
#include <thread>
using namespace std::chrono_literals;

// implementation
#include "vis_00_base.hpp"
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
  return 0;
}