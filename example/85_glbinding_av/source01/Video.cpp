// no preamble
#include <chrono>
#include <iostream>
#include <thread>
void lprint(std::initializer_list<std::string> il, std::string file, int line,
            std::string fun);
extern const std::chrono::time_point<std::chrono::high_resolution_clock>
    g_start_time;
#include "Video.h"
#include <avcpp/codec.h>
#include <avcpp/codeccontext.h>
#include <avcpp/ffmpeg.h>
#include <avcpp/formatcontext.h>
bool Video::GetSuccess() { return success; }
bool Video::Seekable_p() { return ((success) && (ctx.seekable())); }
Video::Video(std::string filename)
    : ctx(av::FormatContext()), fn(filename), success(false) {
  lprint({"open video file", " ", " fn='", fn, "'"}, __FILE__, __LINE__,
         &(__PRETTY_FUNCTION__[0]));
  ctx.openInput(fn, ec);
  if (ec) {
    lprint({"can't open file", " ", " fn='", fn, "'", " ec.message()='",
            ec.message(), "'"},
           __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
    return;
  }
  ctx.findStreamInfo(ec);
  if (ec) {
    lprint(
        {"can't find stream info", " ", " ec.message()='", ec.message(), "'"},
        __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
    return;
  }
  lprint({"stream info", " ", " ctx.seekable()='",
          std::to_string(ctx.seekable()), "'", " ctx.startTime().seconds()='",
          std::to_string(ctx.startTime().seconds()), "'",
          " ctx.duration().seconds()='",
          std::to_string(ctx.duration().seconds()), "'",
          " ctx.streamsCount()='", std::to_string(ctx.streamsCount()), "'"},
         __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
  if (ctx.seekable()) {
    const auto center = (0.50f);
    const auto timeResolution = 100;
    // split second into 100 parts
    ctx.seek(
        {static_cast<long int>(floor(
             ((timeResolution) * (((center) * (ctx.duration().seconds())))))),
         {1, timeResolution}},
        ec);
    if (ec) {
      lprint({"can't seek", " ", " ec.message()='", ec.message(), "'"},
             __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
      return;
    }
  }
  for (size_t i = 0; (i) < (ctx.streamsCount()); i++) {
    auto st = ctx.stream(i);
    if ((AVMEDIA_TYPE_VIDEO) == (st.mediaType())) {
      videoStream = i;
      vst = st;
      break;
    }
  }
  if (vst.isNull()) {
    lprint({"Video stream not found", " "}, __FILE__, __LINE__,
           &(__PRETTY_FUNCTION__[0]));
    return;
  }
  if (vst.isValid()) {
    vdec = av::VideoDecoderContext(vst);
    codec = av::findDecodingCodec(vdec.raw()->codec_id);
    vdec.setCodec(codec);
    vdec.setRefCountedFrames(true);
    vdec.open({{"threads", "1"}}, av::Codec(), ec);
    if (ec) {
      lprint({"can't open codec", " "}, __FILE__, __LINE__,
             &(__PRETTY_FUNCTION__[0]));
      return;
    }
    success = true;
  }
}
av::Packet Video::readPacket() {
  pkt = ctx.readPacket(ec);
  if (ec) {
    lprint({"packet reading error", " ", " ec.message()='", ec.message(), "'"},
           __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
  }
  return pkt;
}
av::VideoFrame Video::decode() {
  auto frame = vdec.decode(pkt, ec);
  if (ec) {
    lprint({"error", " ", " ec.message()='", ec.message(), "'"}, __FILE__,
           __LINE__, &(__PRETTY_FUNCTION__[0]));
  }
  return frame;
}
void Video::seek(float val) {
  const auto timeResolution = 1000;
  if (((success) && (Seekable_p()))) {
    ctx.seek({static_cast<long int>(floor(((timeResolution) * (val)))),
              {1, timeResolution}},
             ec);
    if (ec) {
      lprint({"can't seek", " ", " ec.message()='", ec.message(), "'"},
             __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
      return;
    }
  }
}
float Video::startTime() {
  if (success) {
    return static_cast<float>(ctx.startTime().seconds());
  }
  return (0.f);
}
float Video::duration() {
  if (success) {
    return static_cast<float>(ctx.duration().seconds());
  }
  return (1.0f);
}
Video::~Video() {}