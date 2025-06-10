//
// Created by martin on 6/8/25.
//

// benchmarks/parser_benchmark.cpp
#include <numeric> // For std::iota if needed for varied payload
#include <vector>
#include "benchmark/benchmark.h"
#include "../src/parser.h"
#include "../src/protocol.h" // For forge_serialized_message and constants

// Re-use or redefine the message forging helper for benchmarks
std::vector<unsigned char> forge_bench_message(uint64_t id, uint8_t version,
                                               const std::vector<unsigned char>& payload_data) {
    NetworkProtocol::Message test_msg;
    test_msg.header.id      = id;
    test_msg.header.version = version;
    test_msg.header.length  = payload_data.size();
    test_msg.payload        = payload_data;
    return NetworkProtocol::serializeMessage(test_msg);
}

// Koan: The Swift Scribe - Benchmarking a Single, Small Message
static void BM_ParseSingleSmallMessage(benchmark::State& state) {
    // state.range(0) will be the payload size
    std::vector<unsigned char> payload(state.range(0), 's'); // 's' for small
    std::vector<unsigned char> buffer = forge_bench_message(1, NetworkProtocol::CURRENT_VERSION, payload);

    for (auto _ : state) {
        auto result = Parser::parse_packet(buffer.cbegin(), buffer.cend());
        benchmark::DoNotOptimize(result.message); // Ensure result is not optimized away
        benchmark::DoNotOptimize(result.next_data_iterator);
        if (result.status != Parser::ParseResultStatus::SUCCESS) {
            state.SkipWithError("Benchmark parsing failed for single small message.");
            break;
        }
    }
    state.SetBytesProcessed(static_cast<long>(state.iterations()) * buffer.size());
}
BENCHMARK(BM_ParseSingleSmallMessage)->RangeMultiplier(4)->Range(16, 1024); // Payloads from 16B to 1KB

// Koan: The Endless Stream - Benchmarking Many Small Messages in a Large Buffer
static void BM_ParseMultipleSmallMessages(benchmark::State& state) {
    const int                  num_messages             = state.range(0);    // Number of messages
    const int                  payload_size_per_message = 64;                // Fixed small payload for these messages
    std::vector<unsigned char> small_payload(payload_size_per_message, 'm'); // 'm' for multiple

    std::vector<unsigned char> full_buffer;
    size_t                     single_message_approx_size = NetworkProtocol::Header::SIZE + payload_size_per_message;
    full_buffer.reserve(single_message_approx_size * num_messages);

    for (int i = 0; i < num_messages; ++i) {
        std::vector<unsigned char> current_msg_bytes =
                forge_bench_message(static_cast<uint64_t>(i), NetworkProtocol::CURRENT_VERSION, small_payload);
        full_buffer.insert(full_buffer.end(), current_msg_bytes.begin(), current_msg_bytes.end());
    }

    for (auto _ : state) {
        auto it                    = full_buffer.cbegin();
        auto end_it                = full_buffer.cend();
        int  messages_parsed_count = 0;
        while (it != end_it) {
            // For microbenchmarks, we assume data is mostly valid and sufficient.
            // A NEED_MORE_DATA here means the benchmark setup is flawed for this loop.
            auto result = Parser::parse_packet(it, end_it);
            benchmark::DoNotOptimize(result.message);
            benchmark::DoNotOptimize(result.next_data_iterator);

            if (result.status == Parser::ParseResultStatus::SUCCESS) {
                it = result.next_data_iterator;
                messages_parsed_count++;
            }
            else {
                // This shouldn't happen in a well-formed benchmark buffer
                state.SkipWithError("Benchmark parsing failed or hit NEED_MORE_DATA unexpectedly in multi-message.");
                goto benchmark_loop_exit; // Exit outer loop
            }
        }
        if (messages_parsed_count != num_messages && !state.skipped()) {
            state.SkipWithError("Did not parse all expected messages in multi-message bench.");
        }
    }
benchmark_loop_exit:;
    state.SetBytesProcessed(static_cast<long>(state.iterations()) * full_buffer.size());
    state.counters["Messages"] = benchmark::Counter(static_cast<double>(num_messages), benchmark::Counter::kIsRate);
}
BENCHMARK(BM_ParseMultipleSmallMessages)->RangeMultiplier(2)->Range(8, 128); // 8 to 128 messages

// Koan: The Voluminous Tome - Benchmarking a Single, Large Message
static void BM_ParseSingleLargeMessage(benchmark::State& state) {
    // state.range(0) will be the payload size, e.g., 1KB to 1MB
    std::vector<unsigned char> payload(state.range(0), 'l'); // 'l' for large
    std::vector<unsigned char> buffer = forge_bench_message(1, NetworkProtocol::CURRENT_VERSION, payload);

    for (auto _ : state) {
        auto result = Parser::parse_packet(buffer.cbegin(), buffer.cend());
        benchmark::DoNotOptimize(result.message);
        benchmark::DoNotOptimize(result.next_data_iterator);
        if (result.status != Parser::ParseResultStatus::SUCCESS) {
            state.SkipWithError("Benchmark parsing failed for single large message.");
            break;
        }
    }
    state.SetBytesProcessed(static_cast<long>(state.iterations()) * buffer.size());
}
// Test with payload sizes that might be typical for network packets up to larger chunks.
// e.g. 1400 bytes, 4KB, 16KB, 64KB. Max around 1MB to test large payloads.
BENCHMARK(BM_ParseSingleLargeMessage)->RangeMultiplier(4)->Range(1400, 1024 * 1024);


// Koan: The Beginning of Measurement - Main for Benchmarks
BENCHMARK_MAIN();
