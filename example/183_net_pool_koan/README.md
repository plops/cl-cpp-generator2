i need help with c++ architecture. i have three pools that can contain images (e.g. vector<byte> but the size is constant (128x128) throughout the runtime of the program (the vectors can be allocated for the pool once when the program starts (10 slots should be enough) and the vectors never need to be resized) ) metadata (struct Meta { int i; float a; }, 12 slots are required) or measurement (struct Measurement { double q; double p; }, 20 slots are required)

these 3 types of packages arrive over network as tcp packets (the packet has an u8 indicating id (image id=0, metadata id=1, measurement id=2). a vector on the network has a uint16 prefix indicating the length. i have a  single thread that parses the package and stores it in elements of the three pools and places the elements in three corresponding queues.  (there is ever only one producer thread that fills the 3 queues)

three consumer threads consume the data from their respective queues. consumers only read the data.

create an implementation, use modern c++ and libraries like boost (if that seems reasonable)
use c++20 and prefer modern c++ (jthread)
prefer references to pointers (if possible)

# C++ Multi-Pool Producer-Consumer Network Processor

This project demonstrates a C++ architecture for handling different types of data packets received over a network, processing them concurrently using a producer-consumer pattern with pre-allocated memory pools.

## Problem Solved

Imagine a system receiving a stream of TCP packets containing different data payloads: fixed-size images, metadata structures, and measurement structures. These packets need to be:

1.  **Parsed** to identify the data type and extract the payload.
2.  **Stored efficiently** in memory, avoiding frequent dynamic allocations, especially for large or numerous items like images. The size of each data type (e.g., image dimensions) is known and constant.
3.  **Processed concurrently** by dedicated workers for each data type, ensuring that processing one type doesn't block others.
4.  **Decoupled**, so the network receiving/parsing logic doesn't need to wait for the processing logic, and vice-versa.
5.  **Resource-safe**, ensuring memory buffers are correctly managed and returned for reuse.
6.  **Testable**, allowing individual components (parsing, processing, pool management) to be verified independently.

## Solution Overview

This implementation uses a combination of fixed-size memory pools, thread-safe queues, and modern C++ concurrency features (`std::jthread`, `std::stop_token`) to achieve these goals.

*   **Memory Pools:** Three separate pools are created at startup, one for each data type (Image, Metadata, Measurement). Each pool pre-allocates a fixed number of slots. For `Image`, each `std::vector<std::byte>` within the pool is also pre-sized to the required 128x128 bytes.
*   **Producer Thread:** A single thread simulates receiving network packets. It parses the packet header to identify the type, acquires a free slot (index) from the corresponding pool, copies/deserializes the payload into the pool slot, and then places the *index* of the filled slot onto a dedicated queue for consumers of that data type.
*   **Consumer Threads:** Three separate threads exist, one for each data type. Each consumer waits on its specific queue. When an index arrives, it retrieves a reference to the data in the pool slot associated with that index. It processes the data (read-only) and, upon completion, the pool slot index is automatically returned to the pool's free list, ready for reuse by the producer.
*   **Decoupling:** Thread-safe queues are used to pass indices between the producer and consumers, decoupling their execution. The producer can continue receiving and parsing packets even if a consumer is busy, as long as free pool slots are available. Consumers process data independently.
*   **Resource Management:** A RAII wrapper (`PoolItemReference`) is used by consumers. When a consumer retrieves an item, it gets this wrapper. The wrapper provides access to the data and automatically returns the item's index to the pool's free list when the wrapper goes out of scope, preventing resource leaks.
*   **Testability:** Interfaces (`INetworkReceiver`, `IPoolProducer`, `IPoolConsumer`, `IItemProcessor`) are defined for key components. Dependency Injection is used to provide concrete implementations (or mocks during testing) to the producer and consumer tasks. Google Test and Google Mock are used for unit testing.

## Architecture

```
+---------------------+      Raw Bytes      +-----------------+      Parsed Data + Pool Index     +----------------------+
| Network Interface   |-------------------->| Network Receiver|----------------->| Producer Thread  |--------------------->| Data Pools & Queues  |
| (Simulated/Real)    |                     | (INetworkReceiver)|                 | (Parses, Acquires,|                     | (IPoolProducer/      |
+---------------------+                     +-----------------+                 | Gets Write Buffer,|                     |  IPoolConsumer)      |
| Fills, Submits)   |                     +----------+-----------+
+-----------------+                     |          |           |
| Image    | Meta/Meas |
| Queue    | Queues    |
v          v           v
+-------------------------+      Process      +-------------------+     Pool Index + Ref     +----------------------+   | Item Index (via Queue) |
| Specific Item Processor |<------------------| Consumer Thread   |<------------------------|                      |<--+----------------------+
| (IItemProcessor)        |     (Delegate)    | (Gets Item Ref,   |     (PoolItemReference) |                      |
+-------------------------+                   | Delegates Process)|                         |                      |
+-------------------+                         +----------------------+
(One per type: Image, Meta, Meas)
```

**Key Components:**

*   **`Application`:** Facade class managing the lifecycle (startup, shutdown), creating components, and wiring dependencies.
*   **`INetworkReceiver`:** Interface for receiving raw byte packets.
    *   `SimulatedNetworkReceiver`: Concrete implementation for generating test packets.
*   **`producer_task`:** Function running in the producer `jthread`. Uses `INetworkReceiver` and `IPoolProducer` interfaces.
*   **`IPoolProducer<T>` / `IPoolConsumer<T>`:** Interfaces defining how producers acquire/submit slots and consumers retrieve/return items.
*   **`DataPool<T>`:** Concrete implementation of the pool interfaces. Manages `std::vector<T>` storage, a `ThreadSafeQueue<size_t>` for free indices, and a `ThreadSafeQueue<size_t>` for indices of data ready for consumers.
*   **`PoolItemReference<T>`:** RAII class returned by `consume_item_ref`. Holds a reference to the pool (`IPoolConsumer`) and the item index, ensuring the index is returned via `return_item_index` upon destruction.
*   **`consumer_task<T>`:** Template function running in consumer `jthread`s. Uses `IPoolConsumer` and `IItemProcessor` interfaces.
*   **`IItemProcessor<T>`:** Interface (Strategy pattern) defining the processing logic for a specific data type.
    *   `LoggingImageProcessor`, etc.: Concrete implementations that log processed data.
*   **`ThreadSafeQueue<T>`:** Utility class providing a basic thread-safe queue using `std::mutex` and `std::condition_variable`.

## Key Implementation Details

*   **Modern C++:** Uses C++20 features like `std::jthread`, `std::stop_token`, `std::byte`, `std::span` (where applicable, though index passing is primary here), concepts (potentially), ranges (potentially).
*   **Fixed-Size Pools:** `std::vector<T>` holds the data. `std::vector<std::byte>` for images is pre-sized during pool initialization.
*   **RAII for Pool Items:** `PoolItemReference` ensures pool indices are automatically returned, preventing leaks.
*   **Index Passing:** Lightweight `size_t` indices are passed through queues, not raw pointers or large data objects.
*   **Interfaces & DI:** Core components interact through interfaces, enabling mocking and testing (using Google Test/Mock).
*   **Graceful Shutdown:** Uses `std::jthread::request_stop()`, `std::stop_token`, and `ThreadSafeQueue::stop()` to signal threads and queues to terminate cleanly, allowing threads to finish processing current items if desired.
*   **Serialization (Basic):** Currently uses `memcpy` for simplicity in the simulation. **Note:** A production system would require robust serialization (e.g., Protobuf, FlatBuffers, Cap'n Proto, or manual packing/unpacking respecting endianness and alignment).

## Directory Structure

```
.
├── src/                      # Source code
│   ├── common/               # Basic types, constants, utilities (e.g., ThreadSafeQueue)
│   ├── interfaces/           # Abstract interfaces (IPool*, INetworkReceiver, IItemProcessor)
│   ├── core/                 # Core logic components (DataPool, Producer/Consumer tasks, PoolItemReference)
│   ├── concrete/             # Concrete implementations of interfaces (Simulated Receiver, Logging Processors)
│   └── app/                  # Application facade and main entry point
├── tests/                    # Unit tests
│   ├── mocks/                # Google Mock interfaces
│   └── *.cpp                 # Test files for components (e.g., test_producer.cpp)
└── CMakeLists.txt            # Build configuration file
```

## Building

Requires CMake and a C++20 compliant compiler (GCC >= 10, Clang >= 10). Google Test is fetched automatically via CMake.

```bash
# Navigate to the project root directory
mkdir build
cd build
cmake ..
cmake --build .
```

The application executable (`pool_app`) and test executable (`unit_tests`) will be located in the `build` directory (or a subdirectory like `build/src/app` depending on CMake setup).

## Testing

Google Test is used for unit tests. Mocks are used to isolate components.

```bash
# Navigate to the build directory
cd build

# Run tests using CTest
ctest

# Or run the test executable directly
./unit_tests # Or ./tests/unit_tests depending on exact setup
```

## Dependencies

*   CMake (>= 3.16 recommended)
*   C++20 Compiler (GCC, Clang)
*   GoogleTest (fetched by CMake)
*   pthreads (linked automatically on Linux/macOS via `Threads::Threads`)
```