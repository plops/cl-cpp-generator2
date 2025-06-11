//
// Created by martin on 6/11/25.
//

// In ibuffer.h or a new ibuffer.cpp that includes all buffer headers
#include "buffers/vector_buffer.h"  // (if in ibuffer.cpp)
#include "buffers/deque_buffer.h"
#include "buffers/ring_array_buffer.h"

// Static method in IBuffer class (declaration in .h, definition in .cpp or inline in .h if simple)
/* static */ std::unique_ptr<IBuffer> IBuffer::create(const std::string& type, size_t fixed_capacity) {
    if (type == "VectorBuffer") {
        return std::make_unique<VectorBuffer>();
    } else if (type == "DequeBuffer") {
        return std::make_unique<DequeBuffer>();
    } else if (type == "RingArrayBuffer") {
        return std::make_unique<RingArrayBuffer>(fixed_capacity > 0 ? fixed_capacity : DEFAULT_RING_CAPACITY);
    }
    throw std::runtime_error("Unknown buffer type: " + type);
}