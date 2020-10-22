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
*682 * template <typename T> class client_interface {
public:
  *332 * virtual client_interface();
  *332 * virtual ~client_interface();
  *332 * bool connect(const std::string &host, const uint16_t port);
  *332 * void disconnect();
  *332 * bool is_connected();
  *332 * void send(const message<T> &msg);
  *332 * tsqueue<owned_message<T>> &incoming();

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