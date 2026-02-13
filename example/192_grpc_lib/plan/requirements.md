# Requirements Document

## 1. Functional Requirements
*   **Component Alpha (Client Application):**
    *   Must load a shared processor library dynamically.
    *   Must execute a synchronous "processing session" via the library interface.
    *   Must generate "sensor data" (simulated measurements) in response to library requests.
    *   Must generate log messages and pass them to the library.
*   **Component Beta (Server Application):**
    *   Must host a gRPC server.
    *   Must handle incoming connections from `libtwo`.
    *   Must compute simulation logic: generating timestamps and fusing server-side data with incoming client data.
    *   Must output received log messages from Alpha to the server console.
*   **Library Interface (Shared Abstraction):**
    *   Must define a pure virtual class `IProcessor`.
    *   Must expose a C-compatible factory function (`create_processor`) to ensure ABI stability.
*   **Library One (Local Implementation):**
    *   Implements `IProcessor` completely locally.
    *   Writes logs directly to `std::cout`/`std::cerr`.
    *   Simulates the "Server" logic (timestamps/fusion) internally to mimic the behavior of Libtwo.
*   **Library Two (Remote Implementation):**
    *   Implements `IProcessor` as a gRPC Client.
    *   Forwards computation requests to Component Beta.
    *   Forwards log messages to Component Beta via a dedicated gRPC channel (or metadata).

## 2. Logic & Protocol Requirements (The "Time Sync" Scenario)
*   **The Loop:**
    1.  Library sends a **Server Timestamp** (T_s) to Alpha.
    2.  Alpha generates a **Client Measurement** (M_a) and returns it.
    3.  Library calculates **Result** (R) based on T_s + M_a + (Internal Server Measurement M_b at T_s).
    4.  Alpha receives **Result**.
*   **Termination:** The server/library determines when the session ends and sends a final Summary object.

## 3. Technical & Non-Functional Requirements
*   **Language:** C++20 (targeting GCC 16 capabilities).
*   **Platform:** Gentoo Linux.
*   **Build System:** CMake (3.25+) using `CMakePresets.json`.
*   **Dependencies:** System-installed gRPC and Protobuf; no package managers.
*   **Sanitization:** Debug builds must support ASan (Address), LSan (Leak), and UBSan (Undefined Behavior).
*   **Optimization:** Release builds must use:
    *   `-flto` (Link Time Optimization).
    *   `-g` (External debug symbols).
    *   `-fgraphite-identity -floop-nest-optimize` (Graphite data dependency optimizations).
