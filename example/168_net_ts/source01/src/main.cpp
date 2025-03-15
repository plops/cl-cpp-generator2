// cppcon 2018 vinnie falco Get rich quick! Using Boost.Beast WebSockets and Networking TS
// Networking_in_C++_Part_1_-_MMO_Client_Server_ASIO_Framework_Basics-[2hNdkYInj4g]
#include <array>
// #include <experimental/net>
#include <boost/asio.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <iostream>
#include <string>
#include <thread>

// using namespace std::experimental;
using namespace std;

using namespace boost::asio;

// void cb (error_code ec2, size_t)  { }
int main(int argc, char** argv)
{
    const auto peer{ip::address::from_string("127.0.0.1")};
    const auto port{8080};
    const auto ep{ip::tcp::endpoint(peer, port)};
    io_context ioc;

    auto server = [&]()
    {
        ip::tcp::acceptor acceptor{ioc, ep};
        auto sock{acceptor.accept()};
        constexpr size_t N{1024};
        array<char, N> buffer{};
        mutable_buffer buf{buffer.data(), buffer.size()};
        auto res{sock.read_some(buf)};
        cout << "server: bytesReceived=" << res << endl;
    };
    auto t_server = thread{server};


    auto client = [&]()
    {
        ip::tcp::socket sock(ioc);

        sock.connect(ep);
        if (sock.is_open())
        {
            // const_buffer payload; // pointer and size
            // auto bytesTransferred = sock.write_some(payload);
            // const_buffer header;
            // array<const_buffer, 2> b{header, payload};
            // auto bytesTransferred2 = sock.read_some(b);

            // auto a=net::dynamic_buffer{};
            // auto bytes_transferred=sock.read_some(a.prepare(128)); // prepare read are
            // a.commit(128); // move bytes from write are to read area
            // a.consume(20);

            // sock.write_some(buffer(string("hellow world")));
            sock.async_write_some(buffer(string("hello world"))); //, [](boost::system::error_code& ec, size_t) -> void {});
        }
    };
    auto t_client = thread{client};
    t_client.join();

    auto t_handlers = thread{[&ioc] { ioc.run(); }};
    t_handlers.detach();

    t_server.join();
    t_handlers.join();
}
