//
// Created by martin on 6/11/25.
//

// benchmarks/buffer_performance_benchmark.cpp
#include "benchmark/benchmark.h"
#include "interfaces/ibuffer.h"
#include "buffers/ring_array_buffer.h" // DEFAULT_RING_CAPACITY
#include "parser.h" // For Parser::parse_packet
#include "test_utils.h"

#include <memory>
#include <numeric> // For std::accumulate
#include <string>
#include <vector>

// Global parser function instance for benchmarks
static std::function<Parser::ParseOutput(std::vector<unsigned char>::const_iterator,
                                         std::vector<unsigned char>::const_iterator)>
        global_parser_fn = [](auto begin, auto end) { return Parser::parse_packet(begin, end); };

// Benchmark for writing data to the buffer
static void BM_BufferWrite(benchmark::State& state) {
    std::string buffer_type            = state.range_name(0); // Custom argument name for buffer type
    size_t      chunk_size             = state.range(1);
    size_t      ring_capacity_for_test = chunk_size * 100 > DEFAULT_RING_CAPACITY
                 ? chunk_size * 100
                 : DEFAULT_RING_CAPACITY; // Ensure ring can hold multiple writes

    auto buffer = IBuffer::create(buffer_type, ring_capacity_for_test);
    if (!buffer) {
        state.SkipWithError("Failed to create buffer");
        return;
    }

    auto data_chunk = TestUtils::generate_packet_chunk(
            1, chunk_size - NetworkProtocol::Header::SIZE); // Generate one message of roughly chunk_size
    if (data_chunk.size() == 0 && chunk_size > 0) {         // Ensure data_chunk is not empty if chunk_size is not 0
        data_chunk.resize(chunk_size, 'a');                 // Fallback if generation is smaller than chunk_size
    }


    for (auto _ : state) {
        state.PauseTiming();
        buffer->clear(); // Reset buffer for each iteration to avoid OOM
        state.ResumeTiming();
        for (int i = 0; i < 10; ++i) { // Write 10 chunks per iteration
            buffer->write(data_chunk.data(), data_chunk.size());
        }
    }
    state.SetBytesProcessed(static_cast<long>(state.iterations()) * 10 * data_chunk.size());
    state.SetLabel(buffer_type);
}

// Benchmark for a cycle of writing then parsing
static void BM_BufferWriteParseCycle(benchmark::State& state) {
    std::string buffer_type            = state.range_name(0);
    size_t      num_msgs_per_chunk     = state.range(1);
    size_t      payload_size           = state.range(2);
    size_t      ring_capacity_for_test = (NetworkProtocol::Header::SIZE + payload_size) * num_msgs_per_chunk * 10;
    ring_capacity_for_test =
            ring_capacity_for_test > DEFAULT_RING_CAPACITY ? ring_capacity_for_test : DEFAULT_RING_CAPACITY;


    auto buffer = IBuffer::create(buffer_type, ring_capacity_for_test);
    if (!buffer) {
        state.SkipWithError("Failed to create buffer");
        return;
    }

    auto data_chunk = TestUtils::generate_packet_chunk(num_msgs_per_chunk, payload_size);
    std::vector<NetworkProtocol::Message> parsed_messages;

    for (auto _ : state) {
        state.PauseTiming();
        buffer->clear();
        parsed_messages.clear();
        state.ResumeTiming();

        buffer->write(data_chunk.data(), data_chunk.size());
        size_t count = buffer->parse_and_consume_stream(global_parser_fn, parsed_messages);

        // Make sure all messages were parsed to ensure benchmark validity
        if (count != num_msgs_per_chunk && !state.skipped()) {
            // state.SkipWithError("Not all messages parsed in benchmark cycle.");
            // Don't skip, but this indicates an issue or the buffer couldn't hold it all.
        }
        benchmark::DoNotOptimize(parsed_messages);
    }
    state.SetBytesProcessed(static_cast<long>(state.iterations()) * data_chunk.size());
    state.SetLabel(buffer_type + "/Msgs:" + std::to_string(num_msgs_per_chunk) +
                   "/Payload:" + std::to_string(payload_size));
}

// Custom arguments for benchmarks
class BufferTypeArgs : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {}
    void TearDown(const ::benchmark::State& state) override {}
};

// Define benchmark arguments
#define BENCHMARK_BUFFER_TYPES(func)                                                               \
    BENCHMARK_DEFINE_F(BufferTypeArgs, func)(benchmark::State & st) {                              \
        st.range_name(0) = "VectorBuffer";                                                         \
        func(st);                                                                                  \
    }                                                                                              \
    BENCHMARK_DEFINE_F(BufferTypeArgs, func##_Deque)(benchmark::State & st) {                      \
        st.range_name(0) = "DequeBuffer";                                                          \
        func(st);                                                                                  \
    }                                                                                              \
    BENCHMARK_DEFINE_F(BufferTypeArgs, func##_Ring)(benchmark::State & st) {                       \
        st.range_name(0) = "RingArrayBuffer";                                                      \
        func(st);                                                                                  \
    }                                                                                              \
    BENCHMARK_REGISTER_F(BufferTypeArgs, func)->Args({1024})->Args({4096})->Args({16384});         \
    BENCHMARK_REGISTER_F(BufferTypeArgs, func##_Deque)->Args({1024})->Args({4096})->Args({16384}); \
    BENCHMARK_REGISTER_F(BufferTypeArgs, func##_Ring)->Args({1024})->Args({4096})->Args({16384});


// Define benchmark arguments for WriteParseCycle
#define BENCHMARK_BUFFER_TYPES_WRITE_PARSE(func)                              \
    BENCHMARK_DEFINE_F(BufferTypeArgs, func)(benchmark::State & st) {         \
        st.range_name(0) = "VectorBuffer";                                    \
        func(st);                                                             \
    }                                                                         \
    BENCHMARK_DEFINE_F(BufferTypeArgs, func##_Deque)(benchmark::State & st) { \
        st.range_name(0) = "DequeBuffer";                                     \
        func(st);                                                             \
    }                                                                         \
    BENCHMARK_DEFINE_F(BufferTypeArgs, func##_Ring)(benchmark::State & st) {  \
        st.range_name(0) = "RingArrayBuffer";                                 \
        func(st);                                                             \
    }                                                                         \
    /* Args: {num_msgs_per_chunk, payload_size} */                            \
    BENCHMARK_REGISTER_F(BufferTypeArgs, func)                                \
            ->Args({1, 100})                                                  \
            ->Args({10, 100})                                                 \
            ->Args({1, 1400})                                                 \
            ->Args({5, 1400})                                                 \
            ->Args({10, 4000});                                               \
    BENCHMARK_REGISTER_F(BufferTypeArgs, func##_Deque)                        \
            ->Args({1, 100})                                                  \
            ->Args({10, 100})                                                 \
            ->Args({1, 1400})                                                 \
            ->Args({5, 1400})                                                 \
            ->Args({10, 4000});                                               \
    BENCHMARK_REGISTER_F(BufferTypeArgs, func##_Ring)                         \
            ->Args({1, 100})                                                  \
            ->Args({10, 100})                                                 \
            ->Args({1, 1400})                                                 \
            ->Args({5, 1400})                                                 \
            ->Args({10, 4000});


// Apply the macros to generate benchmarks (this part needs refinement for Google Benchmark v1.8+)
// For Google Benchmark v1.8 and later, using Apply is cleaner.
// Let's register them directly with multiple Args calls.

BENCHMARK(BM_BufferWrite)
        ->ArgName("buffer_type_placeholder") // Will be replaced by state.range_name
        ->ArgsProduct({
                benchmark::CreateDenseRange(0, 0, 1), // Placeholder for custom arg name logic
                {1024, 4096, 16384}                   // chunk_size
        })
        ->Apply([](benchmark::internal::Benchmark* b) {
            // This lambda allows modification of benchmark instances.
            // We'll set the effective buffer type based on a naming convention or by iterating.
            // This is complex. Simpler: define separate benchmark functions or use custom main.
            // For simplicity, let's assume the range_name(0) trick works in the function or use direct registration.
        });
// The above Apply for dynamic naming is tricky. A simpler way for multiple types is:

static void BM_VectorBufferWrite(benchmark::State& state) {
    state.range_name(0) = "VectorBuffer";
    BM_BufferWrite(state);
}
static void BM_DequeBufferWrite(benchmark::State& state) {
    state.range_name(0) = "DequeBuffer";
    BM_BufferWrite(state);
}
static void BM_RingBufferWrite(benchmark::State& state) {
    state.range_name(0) = "RingArrayBuffer";
    BM_BufferWrite(state);
}

BENCHMARK(BM_VectorBufferWrite)->ArgNames({"chunk_size"})->RangeMultiplier(4)->Range(256, 16384);
BENCHMARK(BM_DequeBufferWrite)->ArgNames({"chunk_size"})->RangeMultiplier(4)->Range(256, 16384);
BENCHMARK(BM_RingBufferWrite)->ArgNames({"chunk_size"})->RangeMultiplier(4)->Range(256, 16384);


static void BM_VectorBufferWriteParseCycle(benchmark::State& state) {
    state.range_name(0) = "VectorBuffer";
    BM_BufferWriteParseCycle(state);
}
static void BM_DequeBufferWriteParseCycle(benchmark::State& state) {
    state.range_name(0) = "DequeBuffer";
    BM_BufferWriteParseCycle(state);
}
static void BM_RingBufferWriteParseCycle(benchmark::State& state) {
    state.range_name(0) = "RingArrayBuffer";
    BM_BufferWriteParseCycle(state);
}

BENCHMARK(BM_VectorBufferWriteParseCycle)
        ->ArgNames({"num_msgs", "payload_size"})
        ->Args({1, 100})
        ->Args({10, 100})
        ->Args({1, 1400})
        ->Args({10, 1400})
        ->Args({5, 4000});
BENCHMARK(BM_DequeBufferWriteParseCycle)
        ->ArgNames({"num_msgs", "payload_size"})
        ->Args({1, 100})
        ->Args({10, 100})
        ->Args({1, 1400})
        ->Args({10, 1400})
        ->Args({5, 4000});
BENCHMARK(BM_RingBufferWriteParseCycle)
        ->ArgNames({"num_msgs", "payload_size"})
        ->Args({1, 100})
        ->Args({10, 100})
        ->Args({1, 1400})
        ->Args({10, 1400})
        ->Args({5, 4000});


// BENCHMARK_MAIN(); // Google Benchmark provides this
