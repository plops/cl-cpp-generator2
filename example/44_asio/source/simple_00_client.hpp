#ifndef SIMPLE_00_CLIENT_H
#define SIMPLE_00_CLIENT_H
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
#include "vis_04_client.hpp"
;
// header
#include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
#include "vis_03_connection.hpp"
#include "vis_04_client.hpp"

enum class CustomMsgTypes : uint32_t {
  ServerAccept,
  ServerDeny,
  ServerPing,
  MessageAll,
  ServerMessage
};
;
class CustomClient : public client_interface<CustomMsgTypes> {
public:
  void ping_server();
  void message_all();
};
int main(int argc, char **argv);
#endif