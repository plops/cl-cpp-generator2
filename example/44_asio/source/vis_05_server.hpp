#ifndef VIS_05_SERVER_H
#define VIS_05_SERVER_H
*1079 * #include "utils.h";
*1079 * #include "globals.h";
*1079 * #include<iostream>
#include <chrono>
#include <thread>
    ;
*1079 * #include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
#include "vis_03_connection.hpp"
    ;
*1079 * *682 * template <typename T> class server_interface {
public:
  *332 * virtual server_interface(uint16_t port);
  *332 * virtual ~server_interface();
  *332 * bool start();
  *332 * void stop();
  *332 * void wait_for_client_connection();
  *332 * void message_client(std::shared_ptr<connection<T>> client,
                             const message<T> &msg);
  *332 * void message_all_clients(
             const message<T> &msg,
             std::shared_ptr<connection<T>> ignore_client = nullptr);
  *332 *
      void update(size_t n_max_messages = 0xffffffffffffffff, bool wait = true);

protected:
  *332 * virtual bool on_client_connect(std::shared_ptr<connection<T>> client);
  *332 *
      virtual void on_client_disconnect(std::shared_ptr<connection<T>> client);
  *332 * virtual void on_message(std::shared_ptr<connection<T>> client,
                                 message<T> &msg);
  tsqueue<owned_message<T>> m_q_messages_in;
  std::deque<std::shared_ptr<connection<T>>> m_deq_connections;
  boost::asio::io_context m_asio_context;
  std::thread m_thread_context;
  boost::asio::ip::tcp::acceptor m_asio_acceptor;
  uint32_t n_id_counter = 10000;
};
*1079 * // header
    ;
#endif