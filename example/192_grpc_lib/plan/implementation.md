# Implementation Plan

## Architecture Diagram
```text
[ Application Alpha ]
       |
       v
[ Interface (Header Only) ] <--- defines IProcessor
       |
       +---> [ libone.so ] (Logic: Local CPU)
       |          |
       |          +--> (Console Output)
       |
       +---> [ libtwo.so ] (Logic: gRPC Client)
                  |
             (Network / gRPC)
                  |
                  v
         [ Application Beta (Server) ]
```

## Component Breakdown

### A. The Common Interface (`libcommon`)
A header-only library containing:
*   `IProcessor.hpp`: The pure virtual base class.
*   `DataStructs.hpp`: Plain C++ structs for `Measurement`, `Result`, and `Timestamp`.

### B. The Protocol (`libproto`)
*   `processor.proto`: Defines the gRPC service `RemoteProcessing`.
    *   RPC 1: `BiDiStream` (Bidirectional streaming for the computation loop).
    *   RPC 2: `LogStream` (Client-to-server logging).

### C. Application Beta (The Server)
*   **Role:** The "Truth" source.
*   **Implementation:**
    *   Inherits `RemoteProcessing::Service`.
    *   **Logic:** Maintains a timer. When a client connects, it starts a loop emitting timestamps. When it gets a measurement back, it fuses it with a server-side random noise generator.

### D. Library Two (The Proxy)
*   **Role:** The Adapter.
*   **Implementation:**
    *   `class RemoteProcessor : public IProcessor`.
    *   Translates C++ structs to Protobuf messages.
    *   Manages the gRPC `ClientContext`.
    *   **Logging:** Pushes strings to the `LogStream` RPC.

### E. Library One (The Local Mock)
*   **Role:** The Standalone.
*   **Implementation:**
    *   `class LocalProcessor : public IProcessor`.
    *   **Logic:** Replicates the math of Application Beta exactly, but runs in the calling thread.
    *   **Logging:** `std::clog << msg`.

### F. Application Alpha (The Driver)
*   **Role:** The User.
*   **Implementation:**
    *   Links against the implementation library.
    *   Runs the simulation loop.
