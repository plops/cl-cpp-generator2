//
// Created by martin on 3/29/25.
//

#include "TraceIO.h"
#include <filesystem>
#include <iostream>
using namespace std;
using namespace std::filesystem;

TraceIO::TraceIO(const std::string& uri_) :
    uri{[uri_]() {
        auto file = path(uri_);
        if (!exists(file)) { throw std::runtime_error("file does not exist"); }
        return uri_;
    }()},
    inputStream{uri, ios_base::binary} {}

int TraceIO::write(const uint8_t* data, size_t size) {
    cout << "write " << uri << endl;
    return CustomIO::write(data, size);
}

int TraceIO::read(uint8_t* data, size_t size) {
    cout << "read " << uri << " " << size << endl;
    inputStream.read(reinterpret_cast<char*>(data), size);
    return size;
}
/**
 * ORing this as the "whence" parameter to a seek function causes it to
 * return the filesize without seeking anywhere. Supporting this is optional.
 * If it is not supported then the seek function will return <0.
 */
// #define AVSEEK_SIZE 0x10000

/**
 * Passing this flag as the "whence" parameter to a seek function causes it to
 * seek by any means (like reopening and linear reading) or other normally unreasonable
 * means that can be extremely slow.
 * This may be ignored by the seek code.
 */
// #define AVSEEK_FORCE 0x20000
int64_t TraceIO::seek(int64_t offset, int whence) {
    cout << "seek " << uri << " " << offset << " whence: " << whence << endl;
    ios_base::seekdir seekDir{ios_base::beg};
    switch (whence) {
    case SEEK_CUR:
        seekDir = ios_base::cur;
        break;
    case SEEK_END:
        seekDir = ios_base::end;
        break;
    case SEEK_SET:
        seekDir = ios_base::beg;
        break;
    case AVSEEK_SIZE: {
        auto size{file_size(path(uri))};
        cout << "size " << uri << " " << size << endl;
        return size;
    }
    default:
        cout << "unknown whence " << whence << endl;
        throw std::runtime_error("unsupported whence");
    }
    inputStream.seekg(offset, seekDir);
    return offset;
}

int TraceIO::seekable() const {
    auto res = AVIO_SEEKABLE_NORMAL;
    // res |= AVIO_SEEKABLE_TIME;
    return res;
}

const char* TraceIO::name() const { return "TraceIO"; }
