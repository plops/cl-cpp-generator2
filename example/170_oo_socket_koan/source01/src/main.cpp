#include <ISocket.h>
#include <TCPSocket.h>
#include <iostream>
#include <memory>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    perror("Too few arguments");
    return EXIT_FAILURE;
  }
  string protocolChoice = argv[1];

  unique_ptr<ISocket> currentSocket;

  if (protocolChoice == "tcp") {
    currentSocket = make_unique<TCPSocket>();
  } else if (protocolChoice == "udp") {
    // currentSocket = make_unique<UDPSocket>();
  } else {
    perror("Unknown protocol choice");
    return EXIT_FAILURE;
  }

  cout << "Enter port: " << endl;
  int port;
  cin >> port;

  if (!currentSocket->open(port)) {
    return EXIT_FAILURE;
  }

  string command;
  cout << "Enter command (send/receive/close): " << endl;
  cin >> command;

  if (command == "send") {
    string message;
    cout << "Enter message: " << endl;
    cin >> message;
    if (!currentSocket->send(message)) {
      perror("Send failed");
    } else if (command == "receive") {
      string receivedData = currentSocket->receive();
      if (!receivedData.empty()) {
        cout << "Received: " << receivedData << endl;
      }
    } else if (command != "close") {
      perror("Unknown command");
    }
  }
  currentSocket->close();
  return EXIT_SUCCESS;
}
