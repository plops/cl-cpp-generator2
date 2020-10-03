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
#include <asio.hpp>
#include <asio/ts/buffer.hpp>
#include <asio/ts/internet.hpp>
;
// header;
void grab_some_data (asio::ip::tcp::socket& socket)  ;  
int main (int argc, char** argv)  ;  
#endif