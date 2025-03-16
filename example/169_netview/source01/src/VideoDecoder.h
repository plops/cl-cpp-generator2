//
// Created by martin on 3/16/25.
//

#ifndef VIDEODECODER_H
#define VIDEODECODER_H

// #include <avcpp/ffmpeg.h>
#include <avcpp/formatcontext.h>
#include <avcpp/codec.h>
#include <avcpp/codeccontext.h>
#include <memory>
#include <string>


class VideoDecoder {
  public:
  ~VideoDecoder() = default;
  VideoDecoder() = default;
  /** @brief initialize avformat, start parsing video file
   *
   * @param uri filename for a video file
   * @return success
   */
  bool initialize(const std::string& uri, bool debug = false);
private:
  std::unique_ptr<av::FormatContext> ctx;
  av::Stream vst;
  av::Codec codec;
  std::error_code ec;
  av::VideoDecoderContext vdec;
  av::Packet pkt;
  bool isInitialized{false};
};



#endif //VIDEODECODER_H
