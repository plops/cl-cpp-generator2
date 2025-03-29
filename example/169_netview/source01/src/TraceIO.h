//
// Created by martin on 3/29/25.
//

#ifndef TRACEIO_H
#define TRACEIO_H

#include <avcpp/formatcontext.h>

/*
 * @param opaque An opaque pointer to user-specific data.
 * @param read_packet  A function for refilling the buffer, may be NULL.
 *                     For stream protocols, must never return 0 but rather
 *                     a proper AVERROR code.
 * @param write_packet A function for writing the buffer contents, may be NULL.
 *        The function may not change the input buffers content.
 * @param seek A function for seeking to specified byte position, may be NULL.
 *
 *
    void *opaque,
    int (*read_packet)(void *opaque, uint8_t *buf, int buf_size),
#if FF_API_AVIO_WRITE_NONCONST
    int (*write_packet)(void *opaque, uint8_t *buf, int buf_size),
#else
    int (*write_packet)(void *opaque, const uint8_t *buf, int buf_size),
#endif
    int64_t (*seek)(void *opaque, int64_t offset, int whence));
 */
class TraceIO final : public av::CustomIO {
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
