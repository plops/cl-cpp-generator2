//
// Created by martin on 3/22/25.
//

#include "VideoDecoder.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <unistd.h>

class VideoDecoderBaseTest : public ::testing::Test {
public:
    VideoDecoderBaseTest() {}

protected:
    void SetUp() final {
        auto r=dec.initialize("/home/martin/stage/cl-cpp-generator2/example/143_ryzen_monitor/source01/tests/ring.webm");
        ASSERT_EQ(r, 1);
    }
    void TearDown() final {
        // not needed
    }
    VideoDecoder dec{};
};

TEST_F(VideoDecoderBaseTest, ShortVideo_CollectKeyFrames_CountCorrect) {
    auto kf = dec.collectKeyFrames();
    ASSERT_EQ(kf.size(), 14);
};