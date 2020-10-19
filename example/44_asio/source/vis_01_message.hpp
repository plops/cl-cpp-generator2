#ifndef VIS_01_MESSAGE_H
#define VIS_01_MESSAGE_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
;
#include <boost/asio.hpp>
#include <boost/asio/ts/buffer.hpp>
#include <boost/asio/ts/internet.hpp>
;
template<typename T> class message_header  {
        public:
        T id{};
        uint32_t size = 0;
};
// header
 
template<typename T> class message  {
        public:
        message_header<T> header{};
        std::vector<uint8_t> body;
        size_t size () const   {
                return sizeof(((sizeof(message_header<T>))+(body.size())));
}
        template<typename DataType> friend message<T>& operator<< (message<T>& msg, const DataType& data)    {
                static_assert(std::is_standard_layout<DataType>::value, "data is too complicated");
                        auto i  = msg.body.size();
        msg.body.resize(((msg.body.size())+(sizeof(DataType))));
        std::memcpy(((msg.body.data())+(i)), &data, sizeof(DataType));
                msg.header.size=msg.size();
        return msg;
}
        template<typename DataType> friend message<T>& operator>> (message<T>& msg, DataType& data)    {
                static_assert(std::is_standard_layout<DataType>::value, "data is too complicated");
                        auto i  = ((msg.body.size())-(sizeof(DataType)));
        std::memcpy(&data, ((msg.body.data())+(i)), sizeof(DataType));
        msg.body.resize(i);
                msg.header.size=msg.size();
        return msg;
}
};;
#endif