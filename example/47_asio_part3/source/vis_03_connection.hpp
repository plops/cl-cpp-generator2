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
// header
#include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
template <typename T>
class connection : public std::enable_shared_from_this<connection<T>> {
public:
  enum class owner { server, client };
  connection(owner parent, boost::asio::io_context &asio_context,
             boost::asio::ip::tcp::socket socket,
             tsqueue<owned_message<T>> &q_in)
      : m_socket(std::move(socket)), m_asio_context(asio_context),
        m_q_messages_in(q_in), m_owner_type(parent) {
    if ((m_owner_type) == (owner::server)) {
      m_handshake_out = static_cast<uint64_t>(
          std::chrono::system_clock::now().time_since_epoch().count());
      m_handshake_in = 0;
      m_handshake_check = scramble(m_handshake_out);
    } else {
      m_handshake_in = 0;
      m_handshake_out = 0;
    }
  }
  virtual ~connection() {}
  uint32_t get_id() const { return id; }
  void connect_to_client(uint32_t uid = 0) {
    if ((owner::server) == (m_owner_type)) {
      if (m_socket.is_open()) {
        id = uid;
        read_header();
      }
    }
  }
  void connect_to_server(
      const boost::asio::ip::tcp::resolver::results_type &endpoints) {
    if ((owner::client) == (m_owner_type)) {
      boost::asio::async_connect(
          m_socket, endpoints,
          [this](std::error_code ec, boost::asio::ip::tcp::endpoint endpoint) {
            if (!(ec)) {
              read_header();
            }
          });
    }
  }
  bool disconnect() { return false; }
  bool is_connected() const { return m_socket.is_open(); }
  void send(const message<T> &msg) {
    boost::asio::post(m_asio_context, [this, msg]() {
      auto idle = m_q_messages_out.empty();
      m_q_messages_out.push_back(msg);
      if (idle) {
        write_header();
      }
    });
  }

private:
  void read_header() {
    boost::asio::async_read(
        m_socket,
        boost::asio::buffer(&m_msg_temporary_in.header,
                            sizeof(message_header<T>)),
        [this](std::error_code ec, std::size_t length) {
          if (ec) {

            (std::cout) << (std::setw(10))
                        << (std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count())
                        << (" ") << (std::this_thread::get_id()) << (" ")
                        << (__FILE__) << (":") << (__LINE__) << (" ")
                        << (__func__) << (" ") << ("read header fail") << (" ")
                        << (std::setw(8)) << (" id='") << (id) << ("'")
                        << (std::setw(8)) << (" length='") << (length) << ("'")
                        << (std::setw(8)) << (" sizeof(message_header<T>)='")
                        << (sizeof(message_header<T>)) << ("'") << (std::endl)
                        << (std::flush);
            m_socket.close();
          } else {
            if ((0) < (m_msg_temporary_in.header.size)) {
              m_msg_temporary_in.body.resize(m_msg_temporary_in.size());
              read_body();
            } else {
              add_to_incoming_message_queue();
            }
          }
        });
  }
  void read_body() {
    boost::asio::async_read(
        m_socket,
        boost::asio::buffer(m_msg_temporary_in.body.data(),
                            m_msg_temporary_in.body.size()),
        [this](std::error_code ec, std::size_t length) {
          if (ec) {

            (std::cout) << (std::setw(10))
                        << (std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count())
                        << (" ") << (std::this_thread::get_id()) << (" ")
                        << (__FILE__) << (":") << (__LINE__) << (" ")
                        << (__func__) << (" ") << ("read body fail") << (" ")
                        << (std::setw(8)) << (" id='") << (id) << ("'")
                        << (std::endl) << (std::flush);
            m_socket.close();
          } else {
            add_to_incoming_message_queue();
          }
        });
  }
  void write_header() {
    boost::asio::async_write(
        m_socket,
        boost::asio::buffer(&m_q_messages_out.front().header,
                            sizeof(message_header<T>)),
        [this](std::error_code ec, std::size_t length) {
          if (ec) {

            (std::cout) << (std::setw(10))
                        << (std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count())
                        << (" ") << (std::this_thread::get_id()) << (" ")
                        << (__FILE__) << (":") << (__LINE__) << (" ")
                        << (__func__) << (" ") << ("write header fail") << (" ")
                        << (std::setw(8)) << (" id='") << (id) << ("'")
                        << (std::endl) << (std::flush);
            m_socket.close();
          } else {
            if ((0) < (m_q_messages_out.front().body.size())) {
              write_body();
            } else {
              m_q_messages_out.pop_front();
              if (!(m_q_messages_out.empty())) {
                write_header();
              }
            }
          }
        });
  }
  void write_body() {
    boost::asio::async_write(
        m_socket,
        boost::asio::buffer(m_q_messages_out.front().body.data(),
                            m_q_messages_out.front().body.size()),
        [this](std::error_code ec, std::size_t length) {
          if (ec) {

            (std::cout) << (std::setw(10))
                        << (std::chrono::high_resolution_clock::now()
                                .time_since_epoch()
                                .count())
                        << (" ") << (std::this_thread::get_id()) << (" ")
                        << (__FILE__) << (":") << (__LINE__) << (" ")
                        << (__func__) << (" ") << ("write body fail") << (" ")
                        << (std::setw(8)) << (" id='") << (id) << ("'")
                        << (std::endl) << (std::flush);
            m_socket.close();
          } else {
            m_q_messages_out.pop_front();
            if (!(m_q_messages_out.empty())) {
              write_header();
            }
          }
        });
  }
  void add_to_incoming_message_queue() {
    if ((owner::server) == (m_owner_type)) {
      m_q_messages_in.push_back({this->shared_from_this(), m_msg_temporary_in});
    } else {
      m_q_messages_in.push_back({nullptr, m_msg_temporary_in});
    }
    read_header();
  }
  uint64_t scramble(uint64_t input) {
    // https://youtu.be/hHowZ3bWsio?t=1057
    ;
    auto out = ((input) ^ (0xdeadbeefc0decafellu));
    out = (((((out) & (0xf0f0f0f0f0f0f0llu))) >> (4)) |
           ((((out) & (0x0f0f0f0f0f0f0fllu))) << (4)));
    return ((out) ^ (0xc0deface12345678llu));
  }

protected:
  boost::asio::ip::tcp::socket m_socket;
  boost::asio::io_context &m_asio_context;
  tsqueue<message<T>> m_q_messages_out;
  message<T> m_msg_temporary_in;
  tsqueue<owned_message<T>> &m_q_messages_in;
  owner m_owner_type = owner::server;
  uint32_t id = 0;
  uint64_t m_handshake_out = 0;
  uint64_t m_handshake_in = 0;
  uint64_t m_handshake_check = 0;
};
;
#endif