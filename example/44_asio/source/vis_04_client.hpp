#ifndef VIS_04_CLIENT_H
#define VIS_04_CLIENT_H
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
#include <vis_03_connection.hpp>
;
// header
template<typename T> class client_interface  {
        public:
         connection ()    {
}
         ~connection ()    {
}
        protected:
        boost::asio::io_context m_asio_context;
        private:
        tsqueue<owned_message<T>> m_q_messages_in;
};;
#endif