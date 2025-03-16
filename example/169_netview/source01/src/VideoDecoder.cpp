//
// Created by martin on 3/16/25.
//

#include "VideoDecoder.h"

#include <avcpp/av.h>
#include <format.h>
#include "Histogram.h"
#include <iostream>

using namespace std;

bool VideoDecoder::initialize(const string& uri, bool debug) {
    cout << "Initializing video decoder " << endl;
    auto version    = avformat_version();
    auto versionStr = format("libavformat: {}.{}.{}", AV_VERSION_MAJOR(version), AV_VERSION_MINOR(version),
                             AV_VERSION_MICRO(version));

    cout << versionStr << endl;

    av::init();
    if (debug) { av::setFFmpegLoggingLevel(AV_LOG_DEBUG); }
    ctx = make_unique<av::FormatContext>();
    ctx->openInput(uri, ec);
    if (ec) {
        cerr << "Error opening input file " << uri << " " << ec.message() << endl;
        return false;
    }
    ctx->findStreamInfo(ec);
    if (ec) { cerr << "Error finding stream information " << ec.message() << endl; }
    ssize_t videoStream{-1};
    auto    streamsCount = ctx->streamsCount();
    for (long unsigned int i = 0; i < streamsCount; i++) {
        auto stream = ctx->stream(i);
        auto type   = stream.mediaType();

        if (debug) { cout << "Stream #=" << i << " type=" << type << endl; }
        if (type == AVMEDIA_TYPE_VIDEO) {
            if (debug) { cout << "Found video stream" << endl; }
            vst         = stream;
            videoStream = i;
            break;
        }
    }
    if ((videoStream == -1) || vst.isNull()) {
        cerr << "Video stream not found streamsCount=" << streamsCount << endl;
        return false;
    }
    if (!vst.isValid()) {
        cerr << "Video stream is not valid." << endl;
        return false;
    }
    vdec    = av::VideoDecoderContext(vst);
    auto id = vdec.raw()->codec_id;
    codec   = av::findDecodingCodec(id);
    vdec.setCodec(codec);
    vdec.setRefCountedFrames(true);
    vdec.open({{"threads", "12"}}, av::Codec(), ec);
    if (ec) { cerr << "Error opening video decoder codec id=" << id << " " << ec.message() << endl; }

    auto                 videoPacketCount = 0;
    av::Timestamp        timestamp;
    av::Timestamp        timestamp_prev;
    bool                 prevIsInitialized = false;
    const int            N                 = 32;
    Histogram<double, N> histogram(.0158, .0175);
    auto                 accum = [&](av::Timestamp ts) {
        if (!prevIsInitialized) {
            timestamp_prev    = ts;
            prevIsInitialized = true;
            return;
        }
        auto duration = ts.seconds() - timestamp_prev.seconds();
        histogram.insert(duration);
        timestamp_prev = ts;
    };
    while ((pkt = ctx->readPacket(ec))) {
        if (ec) { cerr << "Packet reading error: " << ec.message() << endl; }
        if (pkt.streamIndex() != videoStream) { continue; }
        timestamp = pkt.ts();
        if (debug)
            accum(timestamp);
        videoPacketCount++;
    }
    if (debug) {
        cout << "Packet #=" << videoPacketCount << " timestamp=" << timestamp.seconds() << "s" <<
                endl;
        cout << "durationRealMin=" << histogram.getObservedMin() << endl;
        cout << "durationRealMax=" << histogram.getObservedMax() << endl;


        for (int i = 0; i < N; i++) {
            cout << "hist "
                    << histogram.getBinX(i) << " "
                    << histogram.getBinY(i) << endl;
        }
    }
    isInitialized = true;
    return true;
}
