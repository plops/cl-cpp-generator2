//
// Created by martin on 3/16/25.
//

#ifndef VIDEODECODER_H
#define VIDEODECODER_H

// #include <avcpp/ffmpeg.h>
#include <avcpp/codec.h>
#include <avcpp/codeccontext.h>
#include <avcpp/formatcontext.h>
#include <memory>
#include <string>


class VideoDecoder {
public:
    ~VideoDecoder() = default;
    VideoDecoder()  = default;
    /** @brief initialize avformat, start parsing video file
     *
     * @param uri filename for a video file
     * @return success
     */
    bool initialize(const std::string& uri, bool debug = false);

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
    };

    void collectKeyFrames();

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
};


#endif // VIDEODECODER_H
