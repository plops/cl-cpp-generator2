#ifndef VIS_05_SERVER_H
#define VIS_05_SERVER_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <thread>
;
#include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
#include "vis_03_connection.hpp"
;
// header
template<typename T> class client_interface  {
        public:
         server_interface (uint16_t port)    : m_socket(m_asio_context){
}
         ~server_interface ()    {
}
        bool start ()    {
}
        void stop ()    {
}
        void wait_for_client_connection ()    {
}
        void message_client (std::shared_ptr<connection<T>> client, const message<T>& msg)    {
}
        void message_all_clients (const message<T>& msg, std::shared_ptr<connection<T>> ignore_client=nullptr)    {
}
        protected:
        bool on_client_connected (std::shared_ptr<connection<T>> client)    {
                return false;
}
        void on_client_disconnected (std::shared_ptr<connection<T>> client)    {
}
        void on_message (std::shared_ptr<connection<T>> client, message<T>& msg)    {
}
        boost::asio::io_context m_asio_context;
        std::thread m_thread_asio;
        boost::asio::ip::tcp::socket m_socket;
        std::unique_ptr<connection<T>> m_connection;
        private:
        tsqueue<owned_message<T>> m_q_messages_in;
};;
#endif