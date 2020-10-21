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
// header
template <typename T> class client_interface {
public:
  client_interface() : m_socket(m_asio_context) {}
  ~client_interface() { disconnect(); }
  bool connect(const std::string &host, const uint16_t port) {
    try {
      auto resolver = boost::asio::ip::tcp::resolver(m_asio_context);
      auto endpoints = resolver.resolve(host, std::to_string(port));
      m_connection = std::make_unique<connection<T>>(
          connection<T>::owner::client, m_asio_context,
          boost::asio::ip::tcp::socket(m_asio_context), m_q_messages_in);
      m_connection->connect_to_server(endpoints);
      m_thread_asio = std::thread([this]() { m_asio_context.run(); });
    } catch (std::exception &e) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("client exception") << (" ") << (std::setw(8))
                  << (" e.what()='") << (e.what()) << ("'") << (std::endl)
                  << (std::flush);
      return false;
    };
    return true;
  }
  void disconnect() {
    if (is_connected()) {
      m_connection->disconnect();
    }
    m_asio_context.stop();
    if (m_thread_asio.joinable()) {
      m_thread_asio.join();
    }
    m_connection.release();
  }
  bool is_connected() {
    if (m_connection) {
      return m_connection->is_connected();
    } else {
      return false;
    }
  }
  void send(const message<T> &msg) {
    if (is_connected()) {
      m_connection->send(msg);
    }
  }
  tsqueue<owned_message<T>> &incoming() { return m_q_messages_in; }

protected:
  boost::asio::io_context m_asio_context;
  std::thread m_thread_asio;
  boost::asio::ip::tcp::socket m_socket;
  std::unique_ptr<connection<T>> m_connection;

private:
  tsqueue<owned_message<T>> m_q_messages_in;
};
;
#endif