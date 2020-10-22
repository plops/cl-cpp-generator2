#ifndef SIMPLE_01_SERVER_H
#define SIMPLE_01_SERVER_H
#include "utils.h"
;
#include "globals.h"
;
#include <chrono>
#include <iostream>
#include <thread>
;
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
;
#include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
#include "vis_03_connection.hpp"
#include "vis_05_server.hpp"
;
// header
#include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
#include "vis_03_connection.hpp"
#include "vis_05_server.hpp"

enum class CustomMsgTypes : uint32_t {
  ServerAccept,
  ServerDeny,
  ServerPing,
  MessageAll,
  ServerMessage
};
;
class CustomServer : public server_interface<CustomMsgTypes> {
public:
  CustomServer(uint16_t port);
  virtual bool
  on_client_connect(std::shared_ptr<connection<CustomMsgTypes>> client);
  virtual void
  on_client_disconnect(std::shared_ptr<connection<CustomMsgTypes>> client);
  virtual void on_message(std::shared_ptr<connection<CustomMsgTypes>> client,
                          message<CustomMsgTypes> &msg);
};
int main(int argc, char **argv);
#endif