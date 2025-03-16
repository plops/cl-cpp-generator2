// Stroustrup A Tour of C++ (2022) p. 151

#include <capnp/ez-rpc.h>
#include <kj/debug.h>

#include <filesystem>
#include <format>
#include <iostream>

#include "VideoArchiveImpl.h"
#include "VideoDecoder.h"


/*
 * Main components
 *
 * Format (Container) - a wrapper, providing sync, metadata and muxing for the
 * streams. Stream - a continuous stream (audio or video) of data over time.
 * Codec - defines how data are enCOded (from Frame to Packet)
 *         and DECoded (from Packet to Frame).
 * Packet - are the data (kind of slices of the stream data) to be decoded as
 * raw frames. Frame - a decoded raw frame (to be encoded or filtered).
 */

using namespace std;
using namespace std::filesystem;

int main(int argc, char* argv[]) {
  string program{argv[0]};
  VideoDecoder decoder;
  decoder.initialize();
  bool isClient = program.find("client") != string_view::npos;

  if (isClient) {
    if (argc < 2) {
      cerr << "Usage: " << program << " SERVER_ADDRESS[:PORT]" << endl;
      return EXIT_FAILURE;
    }
    try {
      capnp::EzRpcClient client(argv[1]);
      auto& waitScope{client.getWaitScope()};
      VideoArchive::Client server = client.getMain<VideoArchive>();
      while (true) {
        cout << "Enter command (list, quit): " << endl;
        string line;
        getline(cin, line);
        stringstream ss(line);
        string command;
        ss >> command;
        if (command == "quit") {
          break;
        } else if (command == "list") {
          auto request = server.getVideoListRequest();
          auto response = request.send().wait(waitScope);
          for (const auto& video : response.getVideoList().getVideos()) {
            cout << video.getSizeBytes() << " " << video.getName().cStr()
                 << endl;
          }
        }
      }

    } catch (const std::exception& e) {
      cerr << e.what() << endl;
      return EXIT_FAILURE;
    }
    return 0;
  }
  // if (argc < 2)
  // {
  //     cerr << "Usage: " << argv[0] << " <path>" << endl;
  //     return EXIT_FAILURE;
  // }
  // path p{argv[1]};

  try {
    capnp::EzRpcServer server(kj::heap<VideoArchiveImpl>(), "localhost:43211");
    auto& waitScope{server.getWaitScope()};
    uint port = server.getPort().wait(waitScope);
    cout << "serving on port " << port << endl;
    kj::NEVER_DONE.wait(waitScope);
  } catch (const std::exception& e) {
    cerr << e.what() << endl;
  }


  //
  // auto ctx = avformat_alloc_context();
  // if (!ctx) {
  //   cerr << "Could not allocate video context" << endl;
  //   return 1;
  // }
  return 0;
}
