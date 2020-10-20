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
         client_interface ()    {
}
         ~client_interface ()    {
                disconnect();
}
        bool connect (const std::string& host, const uint16_t port)    {
                return false;
}
        void disconnect ()    {
}
        bool is_connected_p ()    {
                if ( m_connection ) {
                        return m_connection->is_connected_p();
} else {
                        return false;
}
}
        tsqueue<owned_messages<T>>& incoming ()    {
                return m_q_messages_in;
}
        protected:
        boost::asio::io_context m_asio_context;
        std::thread m_thread_asio;
        boost::asio::ip::tcp::socket m_socket;
        std::unique_ptr<connection<T>> m_connection;
        private:
        tsqueue<owned_message<T>> m_q_messages_in;
};;
#endif