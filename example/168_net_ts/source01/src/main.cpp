// cppcon 2018 vinnie falco Get rich quick! Using Boost.Beast WebSockets and Networking TS

#include <experimental/net>
#include <array>

using namespace std::experimental;

int main(int argc, char** argv)
{
    net::io_context ioc;
    net::ip::tcp::socket sock(ioc);
    net::const_buffer payload;
    auto bytesTransferred = sock.write_some(payload);
    net::const_buffer header;
    std::array<net::const_buffer, 2> b{header, payload};
    auto bytesTransferred2 = sock.read_some(b);

}
