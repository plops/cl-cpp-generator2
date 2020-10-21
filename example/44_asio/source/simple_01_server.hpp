#ifndef SIMPLE_01_SERVER_H
#define SIMPLE_01_SERVER_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
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
#include "simple_01_server.hpp"
;
// header
 
enum class CustomMsgTypes :uint32_t {ServerAccept, ServerDeny, ServerPing, MessageAll, ServerMessage};;
class CustomServer : public server_interface<CustomMsgTypes> {
        public:
         CustomServer (uint16_t port)  ;  
        bool on_client_connect (std::shared_ptr<connection<CustomMsgTypes>> client)  ;  
        void on_client_disconnect (std::shared_ptr<connection<CustomMsgTypes>> client)  ;  
        void on_message (std::shared_ptr<connection<CustomMsgTypes>> client, message<CustomMsgTypes>& msg)  ;  
};
int main (int argc, char** argv)  ;  
#endif