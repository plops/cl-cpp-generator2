#ifndef VIS_00_BASE_H
#define VIS_00_BASE_H
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
;
#include "vis_04_client.hpp"
;
// header
#include "vis_04_client.hpp"
 
enum class CustomMsgTypes :uint32_t {ServerAccept, ServerDeny, ServerPing, MessagesAll, ServerMessage};;
void grab_some_data (boost::asio::ip::tcp::socket& socket)  ;  
int main (int argc, char** argv)  ;  
#endif