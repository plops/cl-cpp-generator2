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
// header
template <typename T> class server_interface {
public:
  server_interface(uint16_t port)
      : m_asio_acceptor(m_asio_context, boost::asio::ip::tcp::endpoint(
                                            boost::asio::ip::tcp::v4(), port)) {
  }
  ~server_interface() { stop(); }
  bool start() {
    try {
      wait_for_client_connection();
      m_thread_context = std::thread([this]() { m_asio_context.run(); });
    } catch (std::exception &e) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("server exception") << (" ") << (std::setw(8))
                  << (" e.what()='") << (e.what()) << ("'") << (std::endl)
                  << (std::flush);
      return false;
    };

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("server started") << (" ") << (std::endl) << (std::flush);
    return true;
  }
  void stop() {
    m_asio_context.stop();
    if (m_thread_context.joinable()) {
      m_thread_context.join();
    }

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("server stopped") << (" ") << (std::endl) << (std::flush);
  }
  void wait_for_client_connection() {
    m_asio_acceptor.async_accept([this](std::error_code ec,
                                        boost::asio::ip::tcp::socket socket) {
      if (ec) {

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("server connection error") << (" ")
                    << (std::setw(8)) << (" ec.message()='") << (ec.message())
                    << ("'") << (std::endl) << (std::flush);
      } else {

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("server new connection") << (" ")
                    << (std::setw(8)) << (" socket.remote_endpoint()='")
                    << (socket.remote_endpoint()) << ("'") << (std::endl)
                    << (std::flush);
        auto newconn = std::make_shared<connection<T>>(
            connection<T>::owner::server, m_asio_context, std::move(socket),
            m_q_messages_in);
        if (on_client_connect(newconn)) {
          m_deq_connections.push_back(std::move(newconn));
          (n_id_counter)++;
          m_deq_connections.back()->connect_to_client(n_id_counter);

          (std::cout) << (std::setw(10))
                      << (std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count())
                      << (" ") << (std::this_thread::get_id()) << (" ")
                      << (__FILE__) << (":") << (__LINE__) << (" ")
                      << (__func__) << (" ") << ("server connection approved")
                      << (" ") << (std::setw(8))
                      << (" m_deq_connections.back()->get_id()='")
                      << (m_deq_connections.back()->get_id()) << ("'")
                      << (std::endl) << (std::flush);
        } else {

          (std::cout) << (std::setw(10))
                      << (std::chrono::high_resolution_clock::now()
                              .time_since_epoch()
                              .count())
                      << (" ") << (std::this_thread::get_id()) << (" ")
                      << (__FILE__) << (":") << (__LINE__) << (" ")
                      << (__func__) << (" ") << ("server connection denied")
                      << (" ") << (std::endl) << (std::flush);
        }
      }
      wait_for_client_connection();
    });
  }
  void message_client(std::shared_ptr<connection<T>> client,
                      const message<T> &msg) {
    if (((client) && (client->is_connected()))) {
      client->send(msg);
    } else {
      on_client_disconnect(client);
      client.reset();
      m_deq_connections.erase(std::remove(m_deq_connections.begin(),
                                          m_deq_connections.end(), client),
                              m_deq_connections.end());
    }
  }
  void
  message_all_clients(const message<T> &msg,
                      std::shared_ptr<connection<T>> ignore_client = nullptr) {
    auto invalid_client_exists = false;
    for (auto &client : m_deq_connections) {
      if (((client) && (client->is_connected()))) {
        if (!((client) == (ignore_client))) {
          client->send(msg);
        }
      } else {
        on_client_disconnect(client);
        client.reset();
        invalid_client_exists = true;
      }
    }
    if (invalid_client_exists) {
      m_deq_connections.erase(std::remove(m_deq_connections.begin(),
                                          m_deq_connections.end(), nullptr),
                              m_deq_connections.end());
    }
  }
  void update(size_t n_max_messages = 0xffffffffffffffff, bool wait = true) {
    if (wait) {
      m_q_messages_in.wait_while_empty();
    }
    auto n_message_count = size_t(0);
    while ((((n_message_count) < (n_max_messages)) &&
            (!(m_q_messages_in.empty())))) {
      auto msg = m_q_messages_in.pop_front();
      on_message(msg.remote, msg.msg);
      (n_message_count)++;
    }
  }

protected:
  bool on_client_connect(std::shared_ptr<connection<T>> client) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("virtual on_client_connect") << (" ") << (std::endl)
                << (std::flush);
    return false;
  }
  void on_client_disconnect(std::shared_ptr<connection<T>> client) {}
  void on_message(std::shared_ptr<connection<T>> client, message<T> &msg) {}
  tsqueue<owned_message<T>> m_q_messages_in;
  std::deque<std::shared_ptr<connection<T>>> m_deq_connections;
  boost::asio::io_context m_asio_context;
  std::thread m_thread_context;
  boost::asio::ip::tcp::acceptor m_asio_acceptor;
  uint32_t n_id_counter = 10000;
};
;
#endif