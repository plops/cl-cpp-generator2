//
// Created by martin on 3/16/25.
//

#ifndef VIDEOARCHIVEIMPL_H
#define VIDEOARCHIVEIMPL_H
#include "proto/video.capnp.h"

class VideoArchiveImpl final : public VideoArchive::Server {
 public:
  VideoArchiveImpl();
  kj::Promise<void> getVideoList(GetVideoListContext context) override;
};

#endif  // VIDEOARCHIVEIMPL_H
