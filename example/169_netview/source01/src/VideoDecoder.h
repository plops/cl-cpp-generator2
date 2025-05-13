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


/**
 * @brief Decodes video files using FFmpeg (via avcpp) to extract stream information and keyframes.
 *
 * This class provides functionalities to initialize a video decoder for a given URI,
 * compute stream statistics, collect information about keyframes, and iterate
 * over packets in the video stream.
 */
class VideoDecoder {
public:
    VideoDecoder() = default;
    /**
     * @brief Initializes the video decoder with the given video file.
     *
     * This method sets up the necessary FFmpeg contexts (format, codec) to
     * read and parse the video file specified by the URI.
     *
     * @param uri The path or URI of the video file to decode.
     * @param traceIO If true, enables tracing of I/O operations (e.g., for custom I/O).
     * @param debug If true, enables additional debug output during initialization.
     * @return True if initialization was successful, false otherwise.
     */
    bool initialize(const std::string& uri, bool traceIO = false, bool debug = false);
    /**
     * @brief Computes and potentially prints statistics about the video stream.
     *
     * This can include information like duration, frame rate, resolution, etc.
     * The exact statistics depend on the underlying FFmpeg/avcpp implementation.
     *
     * @param debug If true, prints the computed statistics to the standard output/error.
     */
    void computeStreamStatistics(bool debug = true);
    /**
     * @brief Structure to hold detailed information about a single keyframe.
     */
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

    /**
     * @brief Collects information for all keyframes in the video stream.
     *
     * This method iterates through the video stream, identifies keyframes,
     * and populates a vector of KeyFrameInfo structures.
     *
     * @return A reference to a vector containing KeyFrameInfo for each keyframe found.
     *         The vector is a member of the class and is populated by this call.
     */
    std::vector<KeyFrameInfo>& collectKeyFrames();

    /**
     * @brief Processes each packet in the video stream using a provided callback function.
     *
     * This method allows for custom processing of each packet. The iteration
     * can be stopped prematurely if the callback returns false.
     *
     * @param callback A function that takes a const av::Packet& and returns a bool.
     *                 The callback should return true to continue processing, or false to stop.
     * @return True if all packets were processed (or the stream ended),
     *         false if the callback requested an early stop.
     */
    bool forEachPacket(function<bool(const av::Packet&)> callback);

private:
    std::unique_ptr<av::FormatContext> ctx;
    av::Stream                         vst;
    av::Codec                          codec;
    std::error_code                    ec;///< Stores error codes from avcpp operations.
    av::VideoDecoderContext            vdec;
    av::Packet                         pkt;
    ssize_t                            videoStream{-1};
    bool                               isInitialized{false};///< Flag indicating if the decoder has been successfully initialized.
    std::vector<KeyFrameInfo>          keyFrames;
    std::unique_ptr<av::CustomIO>      customIO{nullptr};
};


#endif // VIDEODECODER_H
