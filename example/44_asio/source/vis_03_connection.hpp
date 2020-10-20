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
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
;
#include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
;
// header
template<typename T> class connection : public std::enable_shared_from_this<connection<T>> {
        public:
         connection ()    {
}
         ~connection ()    {
}
        bool connect_to_server ()    {
                return false;
}
        bool disconnect ()    {
                return false;
}
        bool is_connected_p () const   {
                return false;
}
        bool send (const message<T>& msg) const   {
                return false;
}
        protected:
        boost::asio::ip::tcp::socket m_socket;
        boost::asio::io_context& m_asio_context;
        tsqueue<message<T>> m_q_messages_out;
};;
#endif