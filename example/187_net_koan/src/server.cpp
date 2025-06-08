//
// Created by martin on 6/8/25.
// This entity listens for connections, receives raw bytes, and feeds them to the parser. It manages a buffer and
// correctly handles partial messages.

// src/server.cpp
#include "parser.h"
#include "protocol.h" // For creating a dummy message to send for testing & Message::toString

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// Linux Socket Headers
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h> // For close()

const int PORT             = 8080;
const int RECV_BUFFER_SIZE = 1024; // How much to try reading from socket at once

// Koan: The Client's Journey - Handling a Single Connection
void handle_client(int client_socket) {
    std::cout << "Tidings of a new traveler (client connected). Socket fd: " << client_socket << std::endl;

    // Koan: The Accumulating Scroll - Our Buffer for Incoming Data
    // This vector will store all bytes received from the client until they are parsed.
    std::vector<unsigned char> data_stream_buffer;

    // Koan: The Mark of Progress - An Iterator to Unparsed Data
    // This iterator always points to the beginning of the data in data_stream_buffer
    // that has not yet been successfully parsed into a complete message.
    // Initially, it's the beginning of our (empty) buffer.
    auto unprocessed_data_iter = data_stream_buffer.cbegin();

    unsigned char temp_recv_buffer[RECV_BUFFER_SIZE];

    try {
        while (true) {
            // Koan: The Act of Receiving - Listening for the Client's Words
            ssize_t bytes_received = recv(client_socket, temp_recv_buffer, RECV_BUFFER_SIZE, 0);

            if (bytes_received < 0) {
                perror("recv failed");
                break;
            }
            if (bytes_received == 0) {
                std::cout << "The traveler has departed (client disconnected)." << std::endl;
                if (unprocessed_data_iter != data_stream_buffer.cend()) {
                    std::cerr << "Warning: Client disconnected with "
                              << std::distance(unprocessed_data_iter, data_stream_buffer.cend())
                              << " unprocessed bytes in buffer." << std::endl;
                }
                break;
            }

            // Koan: Appending Wisdom - Adding New Bytes to Our Scroll
            // Before adding new data, we must carefully manage our iterators.
            // If 'unprocessed_data_iter' is not at the beginning, it means some leading data
            // was already processed and can be removed. Let's find its offset.
            size_t unprocessed_offset = std::distance(data_stream_buffer.cbegin(), unprocessed_data_iter);

            // Add new data to the end of the buffer.
            data_stream_buffer.insert(data_stream_buffer.end(), temp_recv_buffer, temp_recv_buffer + bytes_received);

            // Restore 'unprocessed_data_iter' to point to the same logical data,
            // now possibly in a reallocated buffer.
            unprocessed_data_iter = data_stream_buffer.cbegin() + unprocessed_offset;

            std::cout << "Received " << bytes_received << " bytes. Total buffer size: " << data_stream_buffer.size()
                      << ". Unprocessed starts at offset: " << unprocessed_offset << std::endl;

            // Koan: The Cycle of Interpretation - Parsing Continuously
            // We attempt to parse messages as long as we make progress.
            bool made_progress_this_cycle = true;
            while (made_progress_this_cycle) {
                made_progress_this_cycle = false;

                // If unprocessed_data_iter is at the end, there's nothing left to try parsing from.
                if (unprocessed_data_iter == data_stream_buffer.cend() && !data_stream_buffer.empty()) {
                    // This implies all previous data was consumed.
                    // If data_stream_buffer is not empty here, it's a slight logic issue,
                    // as successful parsing should advance unprocessed_data_iter or the buffer should be cleared.
                    // Let's assume if it's cend, the buffer should be empty or was just cleared.
                }
                if (unprocessed_data_iter == data_stream_buffer.cend() && data_stream_buffer.empty()) {
                    break; // Nothing to parse from an empty buffer.
                }


                // Koan: Invoking the Parser - Seeking a Message
                Parser::ParseOutput result = Parser::parse_packet(unprocessed_data_iter, data_stream_buffer.cend());

                switch (result.status) {
                case Parser::ParseResultStatus::SUCCESS:
                    std::cout << "Parser deciphers a message: " << result.message.value().toString() << std::endl;
                    // Application logic would use result.message.value() here.

                    // The parser has consumed data up to result.next_data_iterator.
                    // We update our main iterator to this new position.
                    unprocessed_data_iter    = result.next_data_iterator;
                    made_progress_this_cycle = true; // We successfully parsed, try again.
                    break;

                case Parser::ParseResultStatus::NEED_MORE_DATA:
                    std::cout << "Parser awaits more fragments: " << result.error_message << std::endl;
                    // No complete message yet. unprocessed_data_iter remains where it was (pointing to the
                    // start of the incomplete segment). We need to break and receive more data.
                    goto end_inner_parse_loop; // Break from the inner while, go to recv

                case Parser::ParseResultStatus::INVALID_DATA:
                    std::cerr << "Parser encounters corrupted script: " << result.error_message << std::endl;
                    if (result.message.has_value()) { // Parser might have salvaged a header for context
                        std::cerr << "  Problematic message shell: " << result.message.value().toString() << std::endl;
                    }
                    // How to recover?
                    // If result.next_data_iterator advanced, we skip the bad part.
                    if (result.next_data_iterator > unprocessed_data_iter) {
                        std::cerr << "  Attempting to skip "
                                  << std::distance(unprocessed_data_iter, result.next_data_iterator)
                                  << " problematic bytes." << std::endl;
                        unprocessed_data_iter    = result.next_data_iterator;
                        made_progress_this_cycle = true; // We skipped, so try parsing again.
                    }
                    else {
                        // Parser could not advance. This is a critical error for this stream.
                        std::cerr << "  Parser cannot recover. Closing connection due to invalid data." << std::endl;
                        close(client_socket);
                        return; // Exit client handler.
                    }
                    break;
                }
            }
        end_inner_parse_loop:;

            // Koan: Pruning the Scroll - Efficient Buffer Management
            // If all data up to 'unprocessed_data_iter' has been fully processed,
            // we can remove it from the beginning of 'data_stream_buffer' to save memory.
            size_t processed_count = std::distance(data_stream_buffer.cbegin(), unprocessed_data_iter);
            if (processed_count > 0) {
                std::cout << "Pruning " << processed_count << " processed bytes from buffer." << std::endl;
                data_stream_buffer.erase(data_stream_buffer.cbegin(), unprocessed_data_iter);
                // After erase, unprocessed_data_iter is invalidated.
                // Since we erased from the beginning up to where it pointed,
                // the new start of unprocessed data is the new beginning of the buffer.
                unprocessed_data_iter = data_stream_buffer.cbegin();
            }
            // If the buffer is now empty, unprocessed_data_iter should correctly be cend().
            if (data_stream_buffer.empty()) {
                unprocessed_data_iter = data_stream_buffer.cend(); // or cbegin(), same for empty
            }

        } // while(true) for recv
    }
    catch (const std::exception &e) {
        std::cerr << "An unexpected trial in client handler: " << e.what() << std::endl;
    }

    // Koan: The Journey's End - Closing the Client Connection
    close(client_socket);
    std::cout << "The channel with traveler (fd: " << client_socket << ") is now closed." << std::endl;
}


int main() {
    // Koan: The Server's Hearth - Preparing to Listen
    int                server_fd;
    struct sockaddr_in address;
    int                opt     = 1;
    socklen_t          addrlen = sizeof(address); // Use socklen_t for accept

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Allow reuse of address and port
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port        = htons(PORT);

    // Binding the socket to the network address and port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // Start listening for incoming connections
    if (listen(server_fd, 3) < 0) { // Listen with a backlog of 3 connections
        perror("Listen failed");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    std::cout << "Server stands ready, listening on port " << PORT << std::endl;
    std::cout << "Send your messages, seeker of wisdom." << std::endl;
    std::cout << "Awaiting travelers..." << std::endl;

    // Koan: The Endless Vigil - Accepting Connections
    // This simple server handles one client at a time.
    while (true) {
        int client_socket;
        if ((client_socket = accept(server_fd, (struct sockaddr *)&address, &addrlen)) < 0) {
            perror("Accept failed");
            // Non-fatal accept error, continue listening.
            continue;
        }
        handle_client(client_socket); // Dedicate attention to this traveler.
    }

    close(server_fd); // Though in this endless loop, this is not reached.
    return 0;
}

/*
Koan: How to Converse with this Server:
1. Compile and run this server program.
2. Use a tool like `netcat` (nc) or write a simple client.

To create a binary message file (e.g., `message.bin`) for testing:
Use the Python script from the previous thought block or a similar utility.
Example: ID=12345, Version=1, Payload="Hello" (length 5)
Serialized:
ID (uint64_t, BE): 00 00 00 00 00 00 30 39
Version (uint8_t): 01
Length (uint64_t, BE): 00 00 00 00 00 00 00 05
Payload ("Hello"): 48 65 6c 6c 6f

Send it: `cat message.bin | nc localhost 8080`
Send multiple: `(cat message1.bin; cat message2.bin) | nc localhost 8080`
Send partial data to test NEED_MORE_DATA, then send the rest.
*/
