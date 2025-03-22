//
// Created by martin on 3/16/25.
//

#include "VideoDecoder.h"
#include <avcpp/av.h>
#include "DurationComputer.h"
// #include <format.h>
#include <iostream>
#include <map>
#include "Histogram.h"


/*
 * Main components
 *
 * Format (Container) - a wrapper, providing sync, metadata and muxing for the
 *                      streams.
 * Stream - a continuous stream (audio or video) of data over time.
 * Codec - defines how data are enCOded (from Frame to Packet)
 *         and DECoded (from Packet to Frame).
 * Packet - are the data (kind of slices of the stream data) to be decoded as
 *          raw frames.
 * Frame - a decoded raw frame (to be encoded or filtered).
 */

using namespace std;

bool VideoDecoder::initialize(const string& uri, bool debug) {
    cout << "Initializing video decoder " << endl;
    auto version    = avformat_version();
    auto versionStr = format("libavformat: {}.{}.{}", AV_VERSION_MAJOR(version), AV_VERSION_MINOR(version),
                             AV_VERSION_MICRO(version));

    cout << versionStr << endl;

    av::init();
    // if (debug) { av::setFFmpegLoggingLevel(AV_LOG_DEBUG); }
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
    vdec.open({{"threads", "1"}}, av::Codec(), ec);
    if (ec) { cerr << "Error opening video decoder codec id=" << id << " " << ec.message() << endl; }

    isInitialized = true;
    return true;
}

void VideoDecoder::computeStreamStatistics(bool debug) {
    // AVFormatContext max_index_size might need to be increased for good seeking in large videos
    //    callback is called during blocking functions to allow abort

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
    Histogram<uint64_t, N> sizeHistogram(0, 10'000);
    Histogram<uint64_t, N> keySizeHistogram(10'000, 100'000);

    vector<uint64_t>      keyPacketNumber;
    vector<av::Timestamp> keyTimes;

    if (debug) { cout << "seekable " << ctx->seekable() << endl; }

    while ((pkt = ctx->readPacket(ec))) {
        if (ec) { cerr << "Packet reading error: " << ec.message() << endl; }
        if (pkt.streamIndex() != videoStream) { continue; }
        auto timestamp = pkt.ts();
        if (debug) {
            auto dur  = packetDuration.insert(timestamp); // 0.017
            auto dur2 = pkt.duration();                   // 16
            cout << "dur=" << dur << " dur2=" << dur2 << endl;
            packetHistogram.insert(dur);
            ptsHistogram.insert(ptsDuration.insert(timestamp));
            dtsHistogram.insert(dtsDuration.insert(timestamp));
            sizeHistogram.insert(pkt.size());
        }
        if (pkt.isKeyPacket()) { keyVideoPacketCount++; }
        else { videoPacketCount++; }
        if (pkt.isComplete()) completePacketCount++;


        if (debug && pkt.isKeyPacket()) {
            av::VideoFrame frame = vdec.decode(pkt, ec);
            if (ec) { cerr << "Error while decoding video frame: " << ec.message() << endl; }
            else if (!frame) { cout << "Empty video frame" << endl; }

            auto pts = frame.pts();

            keyPacketNumber.push_back(videoPacketCount);
            keyTimes.push_back(timestamp);

            auto keyDur = keyDuration.insert(timestamp);
            keyHistogram.insert(keyDur);
            keySizeHistogram.insert(pkt.size());

            clog << "pkt# " << videoPacketCount << "  Frame: " << frame.width() << "x" << frame.height()
                 << ", pktsize=" << pkt.size() << ", size=" << frame.size() << ", tm: " << pts.seconds()
                 << ", tb: " << frame.timeBase() << ", ref=" << frame.isReferenced() << ":" << frame.refCount()
                 << ", key: " << frame.isKeyFrame() << endl;
        }
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
    }
}

std::vector<VideoDecoder::KeyFrameInfo>& VideoDecoder::collectKeyFrames() {
    keyFrames.clear();
    size_t        packetCount   = 0;
    size_t        keyFrameCount = 0;
    av::Timestamp prevTimestamp;
    bool          firstKeyFrame          = true;
    double        timeToPreviousKeyFrame = 0.0;
    // From ffmpeg's packet.h: Packet is reference-counted (pkt->buf is set) and
    // valid indefinitely. The packet must be freed with av_packet_unref() [I guess avcpp handles this] when
    // it is no longer needed. For video, the packet contains exactly one frame.
    while ((pkt = ctx->readPacket(ec))) {
        if (ec) { cerr << "Packet reading error: " << ec.message() << endl; }
        if (pkt.streamIndex() != videoStream) { continue; }
        if (pkt.isKeyPacket()) {
            auto           timestamp = pkt.ts();
            av::VideoFrame frame     = vdec.decode(pkt, ec);
            if (ec) { cerr << "Error while decoding video frame: " << ec.message() << endl; }
            else if (!frame) { cout << "Empty video frame" << endl; }
            else {
                if (firstKeyFrame) { firstKeyFrame = false; }
                else { timeToPreviousKeyFrame = timestamp.seconds() - prevTimestamp.seconds(); }
                prevTimestamp    = timestamp;
                AVPacket     raw = *pkt.raw();
                KeyFrameInfo keyFrameInfo{.timestamp              = timestamp,
                                          .timeToPreviousKeyFrame = timeToPreviousKeyFrame,
                                          .packetIndex            = packetCount,
                                          .frameSize              = frame.size(),
                                          .width                  = frame.width(),
                                          .height                 = frame.height(),
                                          .quality                = frame.quality(),
                                          .bitsPerPixel           = frame.pixelFormat().bitsPerPixel(),
                                          .rawData                = raw.data,
                                          .rawSize                = raw.size};
                keyFrames.push_back(keyFrameInfo);
            }
            keyFrameCount++;
        }
        packetCount++;
    }
    return keyFrames;
}
