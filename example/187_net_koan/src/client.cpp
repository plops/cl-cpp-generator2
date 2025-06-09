//
// Created by martin on 6/9/25.
//

// src/client.cpp
#include "protocol.h" // From your project
#include "test_utils.h" // From the test framework (for create_serialized_message)

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring> // For strerror
#include <cerrno>  // For errno

// Linux Socket Headers
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h> // For close(), sleep()

const char* SERVER_IP = "127.0.0.1"; // localhost
const int SERVER_PORT = 8080;       // Must match server's port

// Helper to send data and handle potential partial sends
bool send_all(int sockfd, const unsigned char* data, size_t length) {
    size_t total_sent = 0;
    while (total_sent < length) {
        ssize_t sent_this_call = send(sockfd, data + total_sent, length - total_sent, 0);
        if (sent_this_call < 0) {
            if (errno == EINTR) continue; // Interrupted by signal, try again
            perror("send failed");
            return false;
        }
        if (sent_this_call == 0) {
            std::cerr << "send returned 0, connection may be closed." << std::endl;
            return false; // Connection closed or issue
        }
        total_sent += sent_this_call;
    }
    return true;
}

// Helper to send a pre-serialized message
bool send_message_bytes(int sockfd, const std::vector<unsigned char>& msg_bytes, const std::string& description) {
    std::cout << "Client: Sending " << description << " (" << msg_bytes.size() << " bytes)..." << std::endl;
    if (!send_all(sockfd, msg_bytes.data(), msg_bytes.size())) {
        std::cerr << "Client: Failed to send all bytes for " << description << std::endl;
        return false;
    }
    std::cout << "Client: Successfully sent " << description << "." << std::endl;
    // Give server a moment to process, especially for fragmented sends or quick succession
    usleep(100000); // 100ms, adjust as needed
    return true;
}


int main() {
    int sock = 0;
    struct sockaddr_in serv_addr;

    // Koan: The Traveler Prepares - Creating the Socket
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("Socket creation error");
        return -1;
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(SERVER_PORT);

    // Koan: Finding the Path - Converting IP Address
    if (inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        close(sock);
        return -1;
    }

    // Koan: The Knock on the Door - Connecting to the Server
    std::cout << "Client: Attempting to connect to server " << SERVER_IP << ":" << SERVER_PORT << "..." << std::endl;
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection Failed");
        close(sock);
        return -1;
    }
    std::cout << "Client: Connected to server!" << std::endl << std::endl;

    try {
        // --- Test Scenario 1: Send a single, complete message ---
        std::vector<unsigned char> payload_s1 = {'S', 'i', 'n', 'g', 'l', 'e'};
        auto msg_bytes_s1 = TestUtils::create_serialized_message(101, NetworkProtocol::CURRENT_VERSION, payload_s1);
        send_message_bytes(sock, msg_bytes_s1, "single complete message (ID 101)");

        // --- Test Scenario 2: Send multiple messages back-to-back (combined in one logical send if small enough, or OS buffers) ---
        std::cout << "\nClient: --- Scenario 2: Multiple messages back-to-back ---" << std::endl;
        std::vector<unsigned char> payload_s2a = {'M', 'u', 'l', 't', 'i', '-', 'A'};
        auto msg_bytes_s2a = TestUtils::create_serialized_message(201, NetworkProtocol::CURRENT_VERSION, payload_s2a);
        send_message_bytes(sock, msg_bytes_s2a, "first of multiple (ID 201)");

        std::vector<unsigned char> payload_s2b = {'M', 'u', 'l', 't', 'i', '-', 'B'};
        auto msg_bytes_s2b = TestUtils::create_serialized_message(202, NetworkProtocol::CURRENT_VERSION, payload_s2b);
        send_message_bytes(sock, msg_bytes_s2b, "second of multiple (ID 202)");

        // --- Test Scenario 3: Send a message in fragments ---
        std::cout << "\nClient: --- Scenario 3: Fragmented message ---" << std::endl;
        std::vector<unsigned char> payload_s3 = {'F', 'r', 'a', 'g', 'm', 'e', 'n', 't', 'e', 'd'};
        auto msg_bytes_s3_full = TestUtils::create_serialized_message(301, NetworkProtocol::CURRENT_VERSION, payload_s3);

        size_t split_point = NetworkProtocol::Header::SIZE + 2; // Send header and 2 bytes of payload
        if (msg_bytes_s3_full.size() <= split_point) split_point = msg_bytes_s3_full.size() / 2;
        if (split_point == 0 && msg_bytes_s3_full.size() > 0) split_point = 1;


        if (split_point > 0 && split_point < msg_bytes_s3_full.size()) {
            std::cout << "Client: Sending fragment 1 (ID 301, " << split_point << " bytes)..." << std::endl;
            if (!send_all(sock, msg_bytes_s3_full.data(), split_point)) {
                throw std::runtime_error("Failed to send fragment 1");
            }
            std::cout << "Client: Sent fragment 1. Sleeping briefly..." << std::endl;
            usleep(200000); // 200ms to give server time to process partial

            std::cout << "Client: Sending fragment 2 (ID 301, " << msg_bytes_s3_full.size() - split_point << " bytes)..." << std::endl;
            if (!send_all(sock, msg_bytes_s3_full.data() + split_point, msg_bytes_s3_full.size() - split_point)) {
                throw std::runtime_error("Failed to send fragment 2");
            }
            std::cout << "Client: Successfully sent fragmented message (ID 301)." << std::endl;
            usleep(100000); // 100ms
        } else {
             std::cout << "Client: Message too small to fragment meaningfully for test, sending whole." << std::endl;
             send_message_bytes(sock, msg_bytes_s3_full, "small message (ID 301) - not fragmented");
        }


        // --- Test Scenario 4: Send a message with an invalid version ---
        std::cout << "\nClient: --- Scenario 4: Invalid version ---" << std::endl;
        uint8_t invalid_version = NetworkProtocol::CURRENT_VERSION + 1;
        if (invalid_version == 0) invalid_version = 2; // handle wrap
        std::vector<unsigned char> payload_s4 = {'B', 'a', 'd', 'V', 'e', 'r'};
        auto msg_bytes_s4 = TestUtils::create_serialized_message(401, invalid_version, payload_s4);
        send_message_bytes(sock, msg_bytes_s4, "message with invalid version (ID 401)");

        // --- Test Scenario 5: Send a message with zero-length payload ---
        std::cout << "\nClient: --- Scenario 5: Zero-length payload ---" << std::endl;
        std::vector<unsigned char> payload_s5_empty; // Empty payload
        auto msg_bytes_s5 = TestUtils::create_serialized_message(501, NetworkProtocol::CURRENT_VERSION, payload_s5_empty);
        send_message_bytes(sock, msg_bytes_s5, "message with zero-length payload (ID 501)");

        // --- Test Scenario 6: Send a larger chunk of multiple messages (like from TestUtils::generate_packet_chunk) ---
        std::cout << "\nClient: --- Scenario 6: Chunk of multiple messages ---" << std::endl;
        // 3 messages, average payload 30 bytes, starting ID 601
        auto chunk_s6 = TestUtils::generate_packet_chunk(3, 30, 601);
        send_message_bytes(sock, chunk_s6, "chunk of 3 messages (IDs 601-603)");


    } catch (const std::exception& e) {
        std::cerr << "Client encountered an error: " << e.what() << std::endl;
    }

    // Koan: The Farewell - Closing the Connection
    std::cout << "\nClient: All test scenarios sent. Closing connection." << std::endl;
    close(sock);

    return 0;
}