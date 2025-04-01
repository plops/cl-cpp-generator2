//
// Created by martin on 3/22/25.
//

#include "VideoDecoder.h"

#include <gtest/gtest.h>
#include <kj/debug.h>
#include <string>

class VideoDecoderBaseTest : public ::testing::Test {
public:
    VideoDecoderBaseTest() {}

protected:
    void SetUp() final {
        cout << "SetUp" << endl << flush;
        dec = make_unique<VideoDecoder>();
    }
    void TearDown() final {
        cout << "TearDown" << endl << flush;
        dec.release();
        dec=nullptr;
    }
    string       videoDir{"/home/martin/stage/cl-cpp-generator2/example/169_netview/source01/tests/"};
    std::unique_ptr<VideoDecoder> dec{nullptr};
};


TEST_F(VideoDecoderBaseTest, ShortVideo_CollectKeyFrames_CountCorrect) {
    auto r = dec->initialize(videoDir + "ring.webm");
    ASSERT_EQ(r, 1);
    auto kf = dec->collectKeyFrames();
    ASSERT_EQ(kf.size(), 14);
};

TEST_F(VideoDecoderBaseTest, OtherShortVideo_CollectKeyFrames_CountCorrect) {
    auto r = dec->initialize(videoDir + "sonic.webm");
    ASSERT_EQ(r, 1);
    auto kf = dec->collectKeyFrames();
    ASSERT_EQ(kf.size(), 21);
};

TEST_F(VideoDecoderBaseTest, WrongVideo_Open_Fails) { ASSERT_FALSE(dec->initialize("/dev/zero")); };

TEST_F(VideoDecoderBaseTest, NonexistingFile_Open_Fails) { ASSERT_FALSE(dec->initialize("/nonexistingfile")); };

TEST_F(VideoDecoderBaseTest, ShortVideo_DecodeFirstPacket_Success) {
    ASSERT_EQ(dec->initialize(videoDir + "ring.webm"), 1);
    int             count = 0;
    vector<uint8_t> buffer;
    auto            cb = [&](const av::Packet& pkt) {
        if (count == 1) return false;
        if (pkt.data()) {
            buffer.resize(pkt.size());
            copy_n(pkt.data(), pkt.size(), buffer.begin());
        }
        count++;
        return true;
    };
    ASSERT_FALSE(dec->forEachPacket(cb));
    KJ_DBG("first packet", count, buffer.size());

    using namespace av;

    error_code          ec;
    VideoDecoderContext vdec;
    Stream              vst;
    auto                codec = av::findDecodingCodec("vp9");
    vdec                      = VideoDecoderContext{codec};
    vdec.open(ec);
    ASSERT_FALSE(ec);
    Packet pkt{buffer.data(), buffer.size(), Packet::wrap_data_static{}};
    auto frame = vdec.decode(pkt);
}

TEST_F(VideoDecoderBaseTest, ShortVideo_TraceCustomIO_Success) {

    ASSERT_EQ(dec->initialize(videoDir + "ring.webm", false), 1);
    int             count = 0;
    vector<uint8_t> buffer;
    auto            cb = [&](const av::Packet& pkt) {
        if (count == 1) return false;
        count++;
        return true;
    };
    ASSERT_FALSE(dec->forEachPacket(cb));
    KJ_DBG("first packet", count, buffer.size());
}



TEST_F(VideoDecoderBaseTest, ShortVideo_CollectKeyFrameData_CountCorrect) {
    auto r = dec->initialize(videoDir + "ring.webm", false);
    ASSERT_EQ(r, 1);
    auto kf = dec->collectKeyFrames();
    ASSERT_EQ(kf.size(), 14);
};