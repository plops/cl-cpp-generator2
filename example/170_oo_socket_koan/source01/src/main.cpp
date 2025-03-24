#include <ISocket.h>
#include <TCPSocket.h>
#include <UDPSocket.h>
#include <iostream>
#include <memory>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    perror("Too few arguments");
    return EXIT_FAILURE;
  }
  const string protocolChoice = argv[1];

  unique_ptr<ISocket> currentSocket;

  if (protocolChoice == "tcp") {
    currentSocket = make_unique<TCPSocket>();
  } else if (protocolChoice == "udp") {
    currentSocket = make_unique<UDPSocket>();
  } else {
    perror("Unknown protocol choice");
    return EXIT_FAILURE;
  }

  cout << "Enter port: " << endl;
  uint16_t port;
  cin >> port;

  if (!currentSocket->open(port)) {
    perror("Failed to open socket");
    return EXIT_FAILURE;
  }

  string command;
  cout << "Enter command (send/receive/close): " << endl;
  cin >> command;

  if (command == "send") {
    string message;
    cout << "Enter message: " << endl;
    cin >> message;
    cout << "Sending: " << message << "..." << endl;
    if (!currentSocket->send(message)) {
      perror("Send failed");
    } else if (command == "receive") {
      if (const string receivedData = currentSocket->receive();
          !receivedData.empty()) {
        cout << "Received: " << receivedData << endl;
      }
    } else if (command != "close") {
      perror("Unknown command");
    }
  }
  currentSocket->close();
  return EXIT_SUCCESS;
}
