// no preamble
#include <spdlog/spdlog.h>

#include <chrono>
#include <iostream>
#include <thread>
extern const std::chrono::time_point<std::chrono::high_resolution_clock>
    g_start_time;
#include <avcpp/codec.h>
#include <avcpp/codeccontext.h>
#include <avcpp/ffmpeg.h>
#include <avcpp/formatcontext.h>

#include "Video.h"
bool Video::GetSuccess() { return success; }
bool Video::Seekable_p() { return (success) & (ctx.seekable()); }
Video::Video(std::string filename)
    : ctx{av::FormatContext()}, fn{filename}, success{false} {
  spdlog::info("open video file  fn='{}'", fn);
  ctx.openInput(fn, ec);
  if (ec) {
    spdlog::info("can't open file  fn='{}'  ec.message()='{}'", fn,
                 ec.message());
    return;
  }
  ctx.findStreamInfo(ec);
  if (ec) {
    spdlog::info("can't find stream info  ec.message()='{}'", ec.message());
    return;
  }
  spdlog::info(
      "stream info  ctx.seekable()='{}'  ctx.startTime().seconds()='{}'  "
      "ctx.duration().seconds()='{}'  ctx.streamsCount()='{}'",
      ctx.seekable(), ctx.startTime().seconds(), ctx.duration().seconds(),
      ctx.streamsCount());
  if (ctx.seekable()) {
    const auto center{0.50F};
    const auto timeResolution{100};
    // split second into 100 parts
    ctx.seek({static_cast<long int>(floor(
                  (timeResolution) * ((center) * (ctx.duration().seconds())))),
              {1, timeResolution}},
             ec);
    if (ec) {
      spdlog::info("can't seek  ec.message()='{}'", ec.message());
      return;
    }
  }
  for ((size_t i) = (0); (i) < (ctx.streamsCount()); i++) {
    auto st{ctx.stream(i)};
    if ((AVMEDIA_TYPE_VIDEO) == (st.mediaType())) {
      (videoStream) = (i);
      (vst) = (st);
      break;
    }
  }
  if (vst.isNull()) {
    spdlog::info("Video stream not found");
    return;
  }
  if (vst.isValid()) {
    (vdec) = (av::VideoDecoderContext(vst));
    (codec) = (av::findDecodingCodec((vdec.raw())->(codec_id)));
    vdec.setCodec(codec);
    vdec.setRefCountedFrames(true);
    vdec.open({{"threads", "1"}}, av::Codec(), ec);
    if (ec) {
      spdlog::info("can't open codec");
      return;
    }
    (success) = (true);
  }
}
av::Packet Video::readPacket() {
  (pkt) = (ctx.readPacket(ec));
  if (ec) {
    spdlog::info("packet reading error  ec.message()='{}'", ec.message());
  }
  return pkt;
}
av::VideoFrame Video::decode() {
  auto frame{vdec.decode(pkt, ec)};
  if (ec) {
    spdlog::info("error  ec.message()='{}'", ec.message());
  }
  return frame;
}
void Video::seek(float val) {
  const auto timeResolution{1000};
  if ((success) & (Seekable_p())) {
    ctx.seek({static_cast<long int>(floor((timeResolution) * (val))),
              {1, timeResolution}},
             ec);
    if (ec) {
      spdlog::info("can't seek  ec.message()='{}'", ec.message());
      return;
    }
  }
}
float Video::startTime() {
  if (success) {
    return static_cast<float>(ctx.startTime().seconds());
  }
  return 0.F;
}
float Video::duration() {
  if (success) {
    return static_cast<float>(ctx.duration().seconds());
  }
  return 1.0F;
}
Video::~Video() {}