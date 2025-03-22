//
// Created by martin on 3/22/25.
//

#include <array>
#include <capnp/ez-rpc.h>
#include <gtest/gtest.h>
#include <kj/debug.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "VideoArchiveImpl.h"

using namespace std;

class VideoArchiveBaseTest : public ::testing::Test {
public:
    VideoArchiveBaseTest() = default;

private:
    void runServer() {
        try {
            capnp::EzRpcServer server(kj::heap<VideoArchiveImpl>(), address);
            auto&              waitScope{server.getWaitScope()};
            uint               port = server.getPort().wait(waitScope);
            cout << "serving on port " << port << endl;
            kj::NEVER_DONE.wait(waitScope);
        }
        catch (const std::exception& e) {
            cerr << "server failure: " << e.what() << endl;
            return;
        }
        catch (...) {
            cerr << "Maybe a server is already running." << endl;
            return;
        }
    }

protected:
    void SetUp() final {
        serverThread = std::thread(&VideoArchiveBaseTest::runServer, this);
        client       = make_unique<capnp::EzRpcClient>{address};
        connection   = make_unique<VideoArchive::Client>(client->getMain<VideoArchive>());
    }
    void         TearDown() final { serverThread.join(); }
    const string address{"localhost:43211"};
    string       videoDir{"/home/martin/stage/cl-cpp-generator2/example/169_netview/source01/tests/"};

    thread                           serverThread{};
    unique_ptr<capnp::EzRpcClient>   client{};
    unique_ptr<VideoArchive::Client> connection{};
};

TEST_F(VideoArchiveBaseTest, StartServerClient_VideoList_ResultCorrect) {
    auto request = connection->getVideoInfoRequest();
    auto videoPath = videoDir + "ring.webm";
    request.setFilePath(videoPath);
    auto& waitScope{client->getWaitScope()};
    auto  response = request.send().wait(waitScope);
    auto videoInfo = response.getVideoInfo();
    ASSERT_EQ(videoInfo.getFilePath(), videoPath);
    ASSERT_EQ(videoInfo.getFileSize(), 1110000);
};
