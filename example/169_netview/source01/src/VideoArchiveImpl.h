//
// Created by martin on 3/16/25.
//

#ifndef VIDEOARCHIVEIMPL_H
#define VIDEOARCHIVEIMPL_H
#include "proto/video.capnp.h"

class VideoArchiveImpl final : public VideoArchive::Server {
public:
    VideoArchiveImpl();
    /**
     * @brief Capnproto RPC call. Will find all video files in a directory.
     * @param context Capnproto boilerplate
     * @return A list of all videofilenames with their sizes in bytes.
     */
    kj::Promise<void> getVideoList(GetVideoListContext context) override;

    kj::Promise<void> getVideoInfo(GetVideoInfoContext context) override;
};

#endif // VIDEOARCHIVEIMPL_H
