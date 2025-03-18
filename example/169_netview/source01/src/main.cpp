// Stroustrup A Tour of C++ (2022) p. 151

#include <capnp/ez-rpc.h>
#include <kj/debug.h>

#include <filesystem>
#include <format>
#include <iostream>

#include "VideoArchiveImpl.h"
#include "VideoDecoder.h"


using namespace std;
using namespace std::filesystem;

int main(int argc, char* argv[]) {
    string       program{argv[0]};
    VideoDecoder decoder;
    bool         isClient = program.find("client") != string_view::npos;

    if (isClient) {
        if (argc < 2) {
            cerr << "Usage: " << program << " SERVER_ADDRESS[:PORT]" << endl;
            return EXIT_FAILURE;
        }
        try {
            cerr << "Client tries to connect to server address: " << argv[1] << endl;
            capnp::EzRpcClient   client(argv[1]);
            auto&                waitScope{client.getWaitScope()};
            VideoArchive::Client server = client.getMain<VideoArchive>();
            while (true) {
                cout << "Enter command (list, quit): " << endl;
                string line;
                getline(cin, line);
                stringstream ss(line);
                string       command;
                ss >> command;
                if (command == "quit") { break; }
                else if (command == "list") {
                    auto   request  = server.getVideoListRequest();
                    auto   response = request.send().wait(waitScope);
                    string selectedFile;
                    int    count = 0;
                    for (const auto& video : response.getVideoList().getVideos()) {
                        cout << video.getSizeBytes() << " " << video.getName().cStr() << endl;
                        if (count == 12) selectedFile = video.getName().cStr();
                        count++;
                    }
                    decoder.initialize(selectedFile, true);
                    decoder.computeStreamStatistics(true);
                }
            }
        }
        catch (const std::exception& e) {
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
        auto&              waitScope{server.getWaitScope()};
        uint               port = server.getPort().wait(waitScope);
        cout << "serving on port " << port << endl;
        kj::NEVER_DONE.wait(waitScope);
    }
    catch (const std::exception& e) {
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
