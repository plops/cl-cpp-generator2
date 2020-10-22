#ifndef VIS_00_BASE_H
#define VIS_00_BASE_H
*1079 * #include "utils.h";
*1079 * #include "globals.h";
*1079 * #include<iostream>
#include <chrono>
#include <thread>
    ;
*1079 * #include<boost / asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
    ;
*1079 * #include "vis_01_message.hpp";
*1079 * #include "vis_04_client.hpp";
*1079 * // header
#include "vis_04_client.hpp"

    enum class CustomMsgTypes : uint32_t {
      ServerAccept,
      ServerDeny,
      ServerPing,
      MessagesAll,
      ServerMessage
    };
;
*1076 * void grab_some_data(boost::asio::ip::tcp::socket &socket);
*1076 * int main(int argc, char **argv);
#endif