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

protected:
    void SetUp() final {
        server     = make_unique<capnp::EzRpcServer>(address);
        auto port =  server->getPort().wait(server->getWaitScope());
        KJ_DBG("Server listening on port: ", port);
        client     = make_unique<capnp::EzRpcClient>(address,port);
        connection = make_unique<VideoArchive::Client>(client->getMain<VideoArchive>());
    }
    void TearDown() final {}

    const string                   address{"localhost"};
    string                         videoDir{"/home/martin/stage/cl-cpp-generator2/example/169_netview/source01/tests/"};
    unique_ptr<capnp::EzRpcServer> server{};
    unique_ptr<capnp::EzRpcClient> client{};
    unique_ptr<VideoArchive::Client> connection{};
};

TEST_F(VideoArchiveBaseTest, StartServerClient_VideoList_ResultCorrect) {
    auto request   = connection->getVideoInfoRequest();
    auto videoPath = videoDir + "sonic.webm";
    request.setFilePath(videoPath);
    auto& waitScope{server->getWaitScope()};
    auto  response  = request.send().wait(waitScope);
    auto  videoInfo = response.getVideoInfo();
    ASSERT_EQ(videoInfo.getFilePath(), videoPath);
    ASSERT_EQ(videoInfo.getFileSize(), 2006194);
};

TEST_F(VideoArchiveBaseTest, StartServerClient_VideoList2_ResultCorrect) {
    auto request   = connection->getVideoInfoRequest();
    auto videoPath = videoDir + "ring.webm";
    request.setFilePath(videoPath);
    auto& waitScope{server->getWaitScope()};
    auto  response  = request.send().wait(waitScope);
    auto  videoInfo = response.getVideoInfo();
    ASSERT_EQ(videoInfo.getFilePath(), videoPath);
    ASSERT_EQ(videoInfo.getFileSize(), 1716911);
};
