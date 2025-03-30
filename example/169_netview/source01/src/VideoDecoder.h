//
// Created by martin on 3/16/25.
//

#ifndef VIDEODECODER_H
#define VIDEODECODER_H

#ifdef NO_LIBS
#include <codec.h>
#include <codeccontext.h>
#include <formatcontext.h>
#else
#include <avcpp/codec.h>
#include <avcpp/codeccontext.h>
#include <avcpp/formatcontext.h>
#endif
#include <memory>
#include <string>

using namespace std;
class VideoDecoder {
public:
    VideoDecoder() = default;
    /** @brief initialize avformat, start parsing video file
     *
     * @param uri filename for a video file
     * @return success
     */
    bool initialize(const std::string& uri, bool traceIO = false, bool debug = false);

    void computeStreamStatistics(bool debug = true);

    struct KeyFrameInfo {
        av::Timestamp timestamp;
        double        timeToPreviousKeyFrame;
        size_t        packetByteOffset;
        size_t        packetIndex;
        size_t        packetSize;
        size_t        frameSize;
        int           width;
        int           height;
        int           quality;
        int           bitsPerPixel;
        uint8_t*      rawData;
        int           rawSize;
    };

    std::vector<KeyFrameInfo>& collectKeyFrames();

    /**
     * \brief Process each packet using the callback
     *
     * \note If the callback returns false, the packet processing will be stopped early
     *
     * \return true if all packets were processed, false if bailed out early
     */
    bool forEachPacket(function<bool(const av::Packet&)> callback);

private:
    std::unique_ptr<av::FormatContext> ctx;
    av::Stream                         vst;
    av::Codec                          codec;
    std::error_code                    ec;
    av::VideoDecoderContext            vdec;
    av::Packet                         pkt;
    ssize_t                            videoStream{-1};
    bool                               isInitialized{false};
    std::vector<KeyFrameInfo>          keyFrames;
    std::unique_ptr<av::CustomIO>      customIO{nullptr};
};


#endif // VIDEODECODER_H
