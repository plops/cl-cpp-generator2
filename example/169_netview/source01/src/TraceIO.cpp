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
    cout << "TraceIO::write " << uri << endl << flush;
    return CustomIO::write(data, size);
}

int TraceIO::read(uint8_t* data, size_t size) {
    cout << "TraceIO::read " << uri << " " << size << endl << flush;
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

// i'm not sure if the following flags can be in whence, i guess not
/* If flags contain AVSEEK_FLAG_BYTE, then all timestamps are in bytes and
 * are the file position (this may not be supported by all demuxers).
 * If flags contain AVSEEK_FLAG_FRAME, then all timestamps are in frames
 * in the stream with stream_index (this may not be supported by all demuxers).
 * Otherwise all timestamps are in units of the stream selected by stream_index
 * or if stream_index is -1, in AV_TIME_BASE units.
 * If flags contain AVSEEK_FLAG_ANY, then non-keyframes are treated as
 * keyframes (this may not be supported by all demuxers).
 * If flags contain AVSEEK_FLAG_BACKWARD, it is ignored.
 */
//
// #define AVSEEK_FLAG_BACKWARD 1 ///< seek backward
// #define AVSEEK_FLAG_BYTE     2 ///< seeking based on position in bytes
// #define AVSEEK_FLAG_ANY      4 ///< seek to any frame, even non-keyframes
// #define AVSEEK_FLAG_FRAME    8 ///< seeking based on frame numberc
int64_t TraceIO::seek(int64_t offset, int whence) {
    whence &= ~AVSEEK_FORCE;
    cout << "TraceIO::seek " << uri << " " << offset << " " << offset/1024/1024 << "MBytes, whence: " << whence << endl << flush;
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
        cout << "unknown whence " << whence << endl << flush;
        return -1;
        // throw std::runtime_error("unsupported whence");
    }
    inputStream.seekg(offset, seekDir);
    return offset;
}

int TraceIO::seekable() const {
    cout << "TraceIO::seekable" << endl << flush;
    auto res = AVIO_SEEKABLE_NORMAL;
   // res |= AVIO_SEEKABLE_TIME;
    return res;
}

const char* TraceIO::name() const { return "TraceIO"; }
