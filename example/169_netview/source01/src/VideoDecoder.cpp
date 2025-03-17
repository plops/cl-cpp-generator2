//
// Created by martin on 3/16/25.
//

#include "VideoDecoder.h"
#include "DurationComputer.h"
#include <avcpp/av.h>
#include <format.h>
#include "Histogram.h"
#include <iostream>
#include <map>

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
    auto streamsCount = ctx->streamsCount();
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

    isInitialized = true;
    return true;
}

void VideoDecoder::computeStreamStatistics(bool debug) {
    auto                   videoPacketCount    = 0;
    auto                   keyVideoPacketCount = 0;
    auto                   completePacketCount = 0;
    const int              N                   = 32;
    Histogram<double, N>   packetHistogram(.0158, .0175);
    DurationComputer       packetDuration;
    Histogram<double, N>   keyHistogram(.0158, 6.5);
    DurationComputer       keyDuration;
    Histogram<double, N>   ptsHistogram(.0158, .0175);
    DurationComputer       ptsDuration;
    Histogram<double, N>   dtsHistogram(.0158, .0175);
    DurationComputer       dtsDuration;
    Histogram<uint64_t, N> sizeHistogram(0, 100'000);
    Histogram<uint64_t, N> keySizeHistogram(0, 100'000);
    Histogram<ptrdiff_t, N>  dataPtrHistogram( -46097531952048,-43123497632562 );

    vector<uint64_t> keyPacketNumber;
    vector<uint8_t*> keyPacketDataPtr;

    while ((pkt = ctx->readPacket(ec))) {
        if (ec) { cerr << "Packet reading error: " << ec.message() << endl; }
        if (pkt.streamIndex() != videoStream) { continue; }
        auto timestamp = pkt.ts();
        if (debug) {
            auto dur = packetDuration.insert(timestamp);
            packetHistogram.insert(dur);
            ptsHistogram.insert(ptsDuration.insert(timestamp));
            dtsHistogram.insert(dtsDuration.insert(timestamp));
            sizeHistogram.insert(pkt.size());

            {
                static uint8_t* previous_data = nullptr;

                auto data = pkt.data();
                if (previous_data) {
                    auto data_gap = std::distance(previous_data, data);
                    dataPtrHistogram.insert(data_gap);
                }
                previous_data = data;
            }
        }
        if (pkt.isKeyPacket()) {
            keyVideoPacketCount++;
            keyPacketNumber.push_back(videoPacketCount);
            keyPacketDataPtr.push_back(pkt.data());
            if (debug) {
                auto keyDur = keyDuration.insert(timestamp);
                keyHistogram.insert(keyDur);
                keySizeHistogram.insert(pkt.size());
            }
        }

        if (pkt.isComplete())
            completePacketCount++;


        videoPacketCount++;
    }
    if (debug) {
        cout << "Packet #=" << videoPacketCount << endl;
        cout << "Complete Packet #=" << completePacketCount << endl;
        cout << "Key #=" << keyVideoPacketCount << endl;
        cout << "packetHistogram " << packetHistogram << endl;
        cout << "keyHistogram " << keyHistogram << endl;
        cout << "sizeHistogram " << sizeHistogram << endl;
        cout << "keySize " << keySizeHistogram << endl;
        cout << "pts " << ptsHistogram << endl;
        cout << "dts " << dtsHistogram << endl;
        cout << "dataPtr " << dataPtrHistogram << endl;
        int i=0;
        for (const auto& e : keyPacketNumber) {
            cout << e << " "
            << reinterpret_cast<int*>(keyPacketDataPtr.at(i))
            // << format("{}",std::format::Ptr(keyPacketDataPtr[i])
                << endl;
            i ++;
        }
    }
}


