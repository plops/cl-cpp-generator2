#ifndef VIS_04_CLIENT_H
#define VIS_04_CLIENT_H
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
template <typename T> class client_interface {
public:
  virtual client_interface();
  virtual ~client_interface();
  bool connect(const std::string &host, const uint16_t port);
  void disconnect();
  bool is_connected();
  void send(const message<T> &msg);
  tsqueue<owned_message<T>> &incoming();

protected:
  boost::asio::io_context m_asio_context;
  std::thread m_thread_asio;
  boost::asio::ip::tcp::socket m_socket;
  std::unique_ptr<connection<T>> m_connection;

private:
  tsqueue<owned_message<T>> m_q_messages_in;
};
// header
;
#endif