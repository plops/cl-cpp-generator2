 
#include "utils.h"
 
#include "globals.h"
 
 
extern State state;
#include <iostream>
#include <chrono>
#include <thread>
#include <deque>
#include <mutex>
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
 
// implementation