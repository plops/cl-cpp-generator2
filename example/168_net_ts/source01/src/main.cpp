// cppcon 2018 vinnie falco Get rich quick! Using Boost.Beast WebSockets and Networking TS

#include <array>
#include <experimental/net>
#include <string>
#include <thread>
#include <iostream>

using namespace std::experimental;
using namespace std;

int main(int argc, char** argv)
{
    error_code ec;
    auto check = [&]()
    {
        if (ec)
            cout<<"error: "<<ec.message()<<endl;
    };
    const auto peer{net::ip::make_address_v4("127.0.0.1",ec)};
    check();
    const auto port{8080};
    const auto ep{net::ip::tcp::endpoint(peer, port)};
    net::io_context ioc;

    auto server = [&]()
    {
        net::ip::tcp::acceptor acceptor{ioc, ep};
        auto sock{acceptor.accept()};
        constexpr size_t N{1024};
        array<char, N> buffer{};
        net::mutable_buffer buf{buffer.data(), buffer.size()};
        auto res{sock.read_some(buf)};
        cout << "server: bytesReceived=" << res << endl;
    };
    auto t_server = thread{server};


    auto client = [&]()
    {
        net::ip::tcp::socket sock(ioc);

        sock.connect(ep,ec);
        check();
        net::const_buffer payload; // pointer and size
        auto bytesTransferred = sock.write_some(payload);
        net::const_buffer header;
        std::array<net::const_buffer, 2> b{header, payload};
        auto bytesTransferred2 = sock.read_some(b);

        // auto a=net::dynamic_buffer{};
        // auto bytes_transferred=sock.read_some(a.prepare(128)); // prepare read are
        // a.commit(128); // move bytes from write are to read area
        // a.consume(20);

        sock.write_some(net::buffer(std::string("hellow world")));
        //
        // sock.async_write_some(net::buffer("hello world"),
        //     [](error_code ec, size_t bytes_transferred)
        //     {
        //       cout << "lambda"  <<"\n";
        //     });
    };
    auto t_client = thread{client};
    t_client.join();

    auto t_handlers = thread{[&ioc]{ioc.run();}};
    t_handlers.detach();

    t_server.join();
    t_handlers.join();
}
