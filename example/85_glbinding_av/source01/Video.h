#ifndef VIDEO_H
#define VIDEO_H

class GLFWwindow;
#include <avcpp/av.h>
#include <avcpp/codec.h>
#include <avcpp/codeccontext.h>
#include <avcpp/ffmpeg.h>
#include <avcpp/formatcontext.h>
class Video {
  std::string fn;
  av::FormatContext ctx;
  av::Stream vst;
  av::Codec codec;
  std::error_code ec;
  av::VideoDecoderContext vdec;
  av::Packet pkt;
  bool success;

 public:
  size_t videoStream = -1;
  bool GetSuccess();
  bool Seekable_p();
  explicit Video(std::string filename);
  av::Packet readPacket();
  av::VideoFrame decode();
  void seek(float val);
  float startTime();
  float duration();
  ~Video();
};

#endif /* !VIDEO_H */