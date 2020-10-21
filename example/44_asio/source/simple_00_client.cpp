
#include "utils.h"

#include "globals.h"

extern State state;
#include "simple_00_client.hpp"
#include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
#include "vis_03_connection.hpp"
#include "vis_04_client.hpp"
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
#include <chrono>
#include <iostream>
#include <thread>
using namespace std::chrono_literals;

// implementation
void CustomClient::ping_server() {
  auto msg = message<CustomMsgTypes>();
  msg.header.id = CustomMsgTypes::ServerPing;
  auto now = std::chrono::system_clock::now();
  (msg) << (now);
  send(msg);
}
void CustomClient::message_all() {
  auto msg = message<CustomMsgTypes>();
  msg.header.id = CustomMsgTypes::MessageAll;
  send(msg);
}
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  auto c = CustomClient();
  c.connect("127.0.0.1", 60000);
  c.ping_server();
  if (c.is_connected()) {
    if (!(c.incoming().empty())) {
      auto msg = c.incoming().pop_front().msg;
      switch (msg.header.id) {
      case CustomMsgTypes::ServerAccept: {

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("server accepted connection") << (" ")
                    << (std::endl) << (std::flush);
        break;
      }
      case CustomMsgTypes::ServerPing: {
        auto now = std::chrono::system_clock::now();
        auto then = now;
        (msg) >> (then);
        auto duration = std::chrono::duration<double>(((now) - (then))).count();

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("ping") << (" ") << (std::setw(8))
                    << (" duration='") << (duration) << ("'") << (std::endl)
                    << (std::flush);
        break;
      }
      case CustomMsgTypes::ServerMessage: {
        uint32_t from_client_id = 0;
        (msg) >> (from_client_id);

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("hello") << (" ") << (std::setw(8))
                    << (" from_client_id='") << (from_client_id) << ("'")
                    << (std::endl) << (std::flush);
        break;
      }
      default: {
        break;
      }
      }
    }
  }
  return 0;
}