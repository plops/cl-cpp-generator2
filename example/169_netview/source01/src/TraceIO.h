//
// Created by martin on 3/29/25.
//

#ifndef TRACEIO_H
#define TRACEIO_H

#include <avcpp/formatcontext.h>

class TraceIO : public av::CustomIO {
public:
    TraceIO() {}
    ~TraceIO() {}
    int         write(const uint8_t* data, size_t size) override;
    int         read(uint8_t* data, size_t size) override;
    int64_t     seek(int64_t offset, int whence) override;
    int         seekable() const override;
    const char* name() const override;
};

#endif // TRACEIO_H
