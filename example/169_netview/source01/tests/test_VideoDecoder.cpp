//
// Created by martin on 3/22/25.
//

#include "VideoDecoder.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <unistd.h>
#include <string>
class VideoDecoderBaseTest : public ::testing::Test {
public:
    VideoDecoderBaseTest() {}

protected:
    void SetUp() final {

    }
    void TearDown() final {
        // not needed
    }
    string videoDir{"/home/martin/stage/cl-cpp-generator2/example/169_netview/source01/tests/"};
    VideoDecoder dec{};
};

TEST_F(VideoDecoderBaseTest, ShortVideo_CollectKeyFrames_CountCorrect) {
    auto r=dec.initialize(videoDir+"ring.webm");
    ASSERT_EQ(r, 1);
    auto kf = dec.collectKeyFrames();
    ASSERT_EQ(kf.size(), 14);
};

TEST_F(VideoDecoderBaseTest, OtherShortVideo_CollectKeyFrames_CountCorrect) {
    auto r=dec.initialize(videoDir+"sonic.webm");
    ASSERT_EQ(r, 1);
    auto kf = dec.collectKeyFrames();
    ASSERT_EQ(kf.size(), 21);
};