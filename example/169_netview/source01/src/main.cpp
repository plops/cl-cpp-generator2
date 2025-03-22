// Stroustrup A Tour of C++ (2022) p. 151

#include <capnp/ez-rpc.h>
#include <kj/debug.h>

#include <filesystem>
#include <format>
#include <iostream>

#include <random>
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
            capnp::EzRpcClient       client(argv[1]);
            auto&                    waitScope{client.getWaitScope()};
            VideoArchive::Client     server = client.getMain<VideoArchive>();
            std::vector<std::string> filenames;
            auto                     request2 = server.getVideoInfoRequest();
            while (true) {
                cout << "Enter command (list, quit, key): " << endl;
                string line;
                getline(cin, line);
                stringstream ss(line);
                string       command;
                ss >> command;
                if (command == "quit") { break; }
                else if (command == "list") {
                    filenames.clear();
                    auto   request  = server.getVideoListRequest();
                    auto   response = request.send().wait(waitScope);
                    string selectedFile;
                    int    count = 0;
                    for (const auto& video : response.getVideoList().getVideos()) {
                        cout << video.getSizeBytes() << " " << video.getName().cStr() << endl;
                        filenames.push_back(video.getName().cStr());
                        if (count == 1) selectedFile = video.getName().cStr();
                        count++;
                    }

                    // decoder.initialize(selectedFile, true);
                    // decoder.computeStreamStatistics(true);
                }
                else if (command == "key") {
                    default_random_engine            generator{};
                    generator.seed(chrono::system_clock::now().time_since_epoch().count());
                    uniform_int_distribution<size_t> distribution(0, filenames.size() - 1);
                    auto rnd    = [&]() { return distribution(generator); };
                    auto choice = rnd();
                    cout << "selected random index: " << choice << endl;
                    if (filenames.size() >= choice) {
                        cout << "selected filename: " << filenames[choice] << endl;
                        request2.setFilePath(filenames[choice]);
                        auto response2 = request2.send().wait(waitScope);
                        auto videoInfo = response2.getVideoInfo();
                        cout << "filename: " << videoInfo.getFilePath().cStr() << endl;
                        cout << "filesize: " << videoInfo.getFileSize() << endl;
                        cout << "Number of keyframes: " << videoInfo.getKeyFrames().size() << endl;
                    }
                    else { cout << "Only {} files" << filenames.size() << endl; }
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
        capnp::EzRpcServer server(kj::heap<VideoArchiveImpl>(), "0.0.0.0:43211");
        auto&              waitScope{server.getWaitScope()};
        uint               port = server.getPort().wait(waitScope);
        cout << "serving on port " << port << endl;
        kj::NEVER_DONE.wait(waitScope);
    }
    catch (const std::exception& e) {
        cerr << "server failure: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        cerr << "Maybe a server is already running." << endl;
        return EXIT_FAILURE;
    }

    return 0;
}
