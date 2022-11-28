// no preamble
;
#include "index.h"
#include "SokolApp.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
std::mutex g_stdout_mutex;