#ifndef VIS_03_CONNECTION_H
#define VIS_03_CONNECTION_H
#include "utils.h"
;
#include "globals.h"
;
#include <chrono>
#include <iostream>
#include <thread>
;
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
;
#include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
;
template <typename T>
class connection : public std::enable_shared_from_this<connection<T>> {
public:
  enum class owner { server, client };
  virtual connection(owner parent, boost::asio::io_context &asio_context,
                     boost::asio::ip::tcp::socket socket,
                     tsqueue<owned_message<T>> &q_in);
  virtual ~connection();
  uint32_t get_id() const;
  void connect_to_client(uint32_t uid = 0);
  void connect_to_server(
      const boost::asio::ip::tcp::resolver::results_type &endpoints);
  bool disconnect();
  bool is_connected() const;
  void send(const message<T> &msg);

private:
  void read_header();
  void read_body();
  void write_header();
  void write_body();
  void add_to_incoming_message_queue();

protected:
  boost::asio::ip::tcp::socket m_socket;
  boost::asio::io_context &m_asio_context;
  tsqueue<message<T>> m_q_messages_out;
  message<T> m_msg_temporary_in;
  tsqueue<owned_message<T>> &m_q_messages_in;
  owner m_owner_type = owner::server;
  uint32_t id = 0;
};
// header
;
#endif