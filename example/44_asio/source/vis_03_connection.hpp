#ifndef VIS_03_CONNECTION_H
#define VIS_03_CONNECTION_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <thread>
;
#include <vis_01_message.hpp>
#include <vis_02_tsqueue.hpp>
;
// header
template<typename T> class connection : public std::enable_shared_from_this<connection<T>>() {
        public:
         connection ()    {
}
         ~connection ()    {
}
        bool connect_to_server ()    {
}
        bool disconnect ()    {
}
        bool is_connected_p () const   {
}
};;
#endif