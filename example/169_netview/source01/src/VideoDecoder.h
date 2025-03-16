//
// Created by martin on 3/16/25.
//

#ifndef VIDEODECODER_H
#define VIDEODECODER_H

#include <avcpp/av.h>
// #include <avcpp/ffmpeg.h>
// #include <avcpp/formatcontext.h>
// #include <avcpp/codec.h>
// #include <avcpp/codeccontext.h>


class VideoDecoder {
  public:
  ~VideoDecoder() = default;
  VideoDecoder() = default;
  void initialize(int width, int height);
private:
  bool m_isInitialized{false};
};



#endif //VIDEODECODER_H
