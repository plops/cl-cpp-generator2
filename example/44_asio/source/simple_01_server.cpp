
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
#include "simple_01_server.hpp"
CustomServer::CustomServer(uint16_t port)
    : server_interface<CustomMsgTypes>(port) {}
bool CustomServer::on_client_connect(
    std::shared_ptr<connection<CustomMsgTypes>> client) {
  auto msg = message<CustomMsgTypes>();
  msg.header.id = CustomMsgTypes::ServerAccept;
  client->send(msg);
  return true;
}
void CustomServer::on_client_disconnect(
    std::shared_ptr<connection<CustomMsgTypes>> client) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("removing") << (" ")
      << (std::setw(8)) << (" client->get_id()='") << (client->get_id())
      << ("'") << (std::endl) << (std::flush);
}
void CustomServer::on_message(
    std::shared_ptr<connection<CustomMsgTypes>> client,
    message<CustomMsgTypes> &msg) {
  switch (msg.header.id) {
  case CustomMsgTypes::ServerPing: {
    client->send(msg);

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("ping") << (" ") << (std::setw(8))
                << (" client->get_id()='") << (client->get_id()) << ("'")
                << (std::endl) << (std::flush);
    break;
  }
  case CustomMsgTypes::MessageAll: {
    auto msg = message<CustomMsgTypes>();
    msg.header.id = CustomMsgTypes::ServerMessage;
    (msg) << (client->get_id());
    message_all_clients(msg, client);

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("message all") << (" ") << (std::setw(8))
                << (" client->get_id()='") << (client->get_id()) << ("'")
                << (std::endl) << (std::flush);
    break;
  }
  default: {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("unsupported message") << (" ") << (std::endl)
                << (std::flush);
    break;
  }
  }
}
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  auto server = CustomServer(60000);
  server.start();
  while (true) {
    server.update(-1, true);
  }
  return 0;
}