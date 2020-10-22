#ifndef VIS_05_SERVER_H
#define VIS_05_SERVER_H
#include "utils.h"
;
#include "globals.h"
;
#include <chrono>
#include <iostream>
#include <thread>
;
#include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
#include "vis_03_connection.hpp"
;
template <typename T> class server_interface {
public:
  virtual server_interface(uint16_t port);
  virtual ~server_interface();
  bool start();
  void stop();
  void wait_for_client_connection();
  void message_client(std::shared_ptr<connection<T>> client,
                      const message<T> &msg);
  void
  message_all_clients(const message<T> &msg,
                      std::shared_ptr<connection<T>> ignore_client = nullptr);
  void update(size_t n_max_messages = 0xffffffffffffffff, bool wait = true);

protected:
  virtual bool on_client_connect(std::shared_ptr<connection<T>> client);
  virtual void on_client_disconnect(std::shared_ptr<connection<T>> client);
  virtual void on_message(std::shared_ptr<connection<T>> client,
                          message<T> &msg);
  tsqueue<owned_message<T>> m_q_messages_in;
  std::deque<std::shared_ptr<connection<T>>> m_deq_connections;
  boost::asio::io_context m_asio_context;
  std::thread m_thread_context;
  boost::asio::ip::tcp::acceptor m_asio_acceptor;
  uint32_t n_id_counter = 10000;
};
// header
;
#endif