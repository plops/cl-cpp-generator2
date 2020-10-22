#ifndef VIS_03_CONNECTION_H
#define VIS_03_CONNECTION_H
*1079 * #include "utils.h";
*1079 * #include "globals.h";
*1079 * #include<iostream>
#include <chrono>
#include <thread>
    ;
*1079 * #include<boost / asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
    ;
*1079 * #include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
    ;
*1079 * *682 *
    template <typename T>
    class connection : public std::enable_shared_from_this<connection<T>> {
public:
  enum class owner { server, client };
  *332 * virtual connection(owner parent, boost::asio::io_context &asio_context,
                            boost::asio::ip::tcp::socket socket,
                            tsqueue<owned_message<T>> &q_in);
  *332 * virtual ~connection();
  *332 * uint32_t get_id() const;
  *332 * void connect_to_client(uint32_t uid = 0);
  *332 * void connect_to_server(
             const boost::asio::ip::tcp::resolver::results_type &endpoints);
  *332 * bool disconnect();
  *332 * bool is_connected() const;
  *332 * void send(const message<T> &msg);

private:
  *332 * void read_header();
  *332 * void read_body();
  *332 * void write_header();
  *332 * void write_body();
  *332 * void add_to_incoming_message_queue();

protected:
  boost::asio::ip::tcp::socket m_socket;
  boost::asio::io_context &m_asio_context;
  tsqueue<message<T>> m_q_messages_out;
  message<T> m_msg_temporary_in;
  tsqueue<owned_message<T>> &m_q_messages_in;
  owner m_owner_type = owner::server;
  uint32_t id = 0;
};
*1079 * // header
    ;
#endif