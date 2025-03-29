//
// Created by martin on 3/29/25.
//

#include "TraceIO.h"
#include <iostream>
using namespace std;

int TraceIO::write(const uint8_t* data, size_t size) {
    cout << "write" << endl;
    return CustomIO::write(data, size);
}
int         TraceIO::read(uint8_t* data, size_t size) { return CustomIO::read(data, size); }
int64_t     TraceIO::seek(int64_t offset, int whence) { return CustomIO::seek(offset, whence); }
int         TraceIO::seekable() const { return CustomIO::seekable(); }
const char* TraceIO::name() const { return CustomIO::name(); }
