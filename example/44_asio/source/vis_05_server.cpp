
#include "utils.h"

#include "globals.h"

extern State state;
#include "vis_01_message.hpp"
#include "vis_02_tsqueue.hpp"
#include "vis_03_connection.hpp"
#include <chrono>
#include <iostream>
#include <thread>

// implementation
*707 * template <typename T>
       * 332 *
    server_interface::server_interface(uint16_t port)
    : m_asio_acceptor(m_asio_context, boost::asio::ip::tcp::endpoint(
                                          boost::asio::ip::tcp::v4(), port)) *
      411 * {} * 707 * template <typename T>
                       * 332 * server_interface::~server_interface() * 411 * {
  stop();
}
*707 * template <typename T> * 332 * bool server_interface::start() * 411 * {
  try {
    wait_for_client_connection();
    m_thread_context = std::thread([this]() { m_asio_context.run(); });
  } catch (std::exception &e) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("server exception") << (" ") << (std::setw(8))
                << (" e.what()='") << (e.what()) << ("'") << (std::endl)
                << (std::flush);
    return false;
  };

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("server started")
      << (" ") << (std::endl) << (std::flush);
  return true;
}
*707 * template <typename T> * 332 * void server_interface::stop() * 411 * {
  m_asio_context.stop();
  if (m_thread_context.joinable()) {
    m_thread_context.join();
  }

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("server stopped")
      << (" ") << (std::endl) << (std::flush);
}
*707 * template <typename T>
       * 332 * void server_interface::wait_for_client_connection() * 411 * {
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
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("server connection approved") << (" ")
                    << (std::setw(8))
                    << (" m_deq_connections.back()->get_id()='")
                    << (m_deq_connections.back()->get_id()) << ("'")
                    << (std::endl) << (std::flush);
      } else {

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("server connection denied") << (" ")
                    << (std::endl) << (std::flush);
      }
    }
    wait_for_client_connection();
  });
}
*707 * template <typename T>
       * 332 *
    void server_interface::message_client(std::shared_ptr<connection<T>> client,
                                          const message<T> &msg) *
    411 * {
  if (((client) && (client->is_connected()))) {
    client->send(msg);
  } else {
    on_client_disconnect(client);
    client.reset();
    m_deq_connections.erase(
        std::remove(m_deq_connections.begin(), m_deq_connections.end(), client),
        m_deq_connections.end());
  }
}
*707 * template <typename T>
       * 332 *
    void server_interface::message_all_clients(
        const message<T> &msg,
        std::shared_ptr<connection<T>> ignore_client = nullptr) *
    411 * {
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
*707 * template <typename T>
       * 332 *
    void server_interface::update(size_t n_max_messages = 0xffffffffffffffff,
                                  bool wait = true) *
    411 * {
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
*707 * template <typename T>
       * 332 *
    bool
    server_interface::on_client_connect(std::shared_ptr<connection<T>> client) *
    411 * {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ")
      << ("virtual on_client_connect") << (" ") << (std::endl) << (std::flush);
  return false;
}
*707 * template <typename T>
       * 332 *
    void server_interface::on_client_disconnect(
        std::shared_ptr<connection<T>> client) *
    411 * {} * 707 * template <typename T>
                     * 332 *
    void server_interface::on_message(std::shared_ptr<connection<T>> client,
                                      message<T> &msg) *
    411 * {}