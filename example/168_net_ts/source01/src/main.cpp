// cppcon 2018 vinnie falco Get rich quick! Using Boost.Beast WebSockets and Networking TS

#include <array>
#include <experimental/net>
#include <string>
#include <thread>

using namespace std::experimental;
using namespace std;

int main(int argc, char** argv)
{
    const auto peer{net::ip::make_address_v4("127.0.0.1")};
    const auto port{8080};
    const auto ep{net::ip::tcp::endpoint(peer, port)};
    auto server = [&]()
    {
        net::io_context ioc;
        net::ip::tcp::acceptor acceptor{ioc, ep};
        auto sock{acceptor.accept()};
        constexpr size_t N{1024};
        array<char, N> buffer{};
        net::mutable_buffer buf{buffer.data(), buffer.size()};
        auto res{sock.read_some(buf)};
    };
    auto t_server = thread{server};

    auto client = [&]()
    {
        net::io_context ioc;
        net::ip::tcp::socket sock(ioc);

        sock.connect(ep);
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
    };
    auto t_client = thread{client};
    t_client.join();
    t_server.join();
}
