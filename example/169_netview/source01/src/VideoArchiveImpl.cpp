//
// Created by martin on 3/16/25.
//

#include "VideoArchiveImpl.h"

#include <capnp/message.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <regex>

#include "VideoDecoder.h"

using namespace std;
using namespace std::filesystem;

VideoArchiveImpl::VideoArchiveImpl() = default;

kj::Promise<void> VideoArchiveImpl::getVideoInfo(GetVideoInfoContext context) {
    cout << "VideoArchiveImpl::getVideoInfo: " << endl;

    VideoDecoder decoder;
    auto filename = context.getParams().getFilePath();

    // Look at the file
    auto   builder   = kj::heap<capnp::MallocMessageBuilder>();
    auto   root      = builder->initRoot<VideoInfo>();
    root.setFilePath(filename);
    auto fileSize = file_size(path(filename));
    root.setFileSize(fileSize);
    context.getResults().setVideoInfo(root); // set information in case avcpp exits early

    // Parse the video
    decoder.initialize(filename);
    auto keyFrames = decoder.collectKeyFrames();
    if (keyFrames.size() == 0) {
        cerr << "VideoArchiveImpl::getVideoInfo: No keyframes found" << endl;
        return kj::READY_NOW;
    }

    // Process keyframes
    auto   keyFrameList = root.initKeyFrames(keyFrames.size());
    size_t i         = 0;
    for (const auto& kf : keyFrames) {
        auto elementBuilder = kj::heap<capnp::MallocMessageBuilder>();
        auto elementRoot    = elementBuilder->initRoot<KeyFrame>();
        auto ts = kf.timestamp;
        auto ts_value = ts.timestamp();
        elementRoot.setTimePosition(ts_value);
        auto timebase=ts.timebase();
        auto timebaseTop = timebase.getNumerator();
        auto timebaseBottom = timebase.getDenominator();
        auto timebaseBuilder = kj::heap<capnp::MallocMessageBuilder>();
        auto timebaseRoot = timebaseBuilder->initRoot<Rational>();
        timebaseRoot.setTop(timebaseTop);
        timebaseRoot.setBottom(timebaseBottom);
        elementRoot.setTimebase(timebaseRoot);
        elementRoot.setDurationSincePreviousKeyframe(kf.timeToPreviousKeyFrame);
        elementRoot.setPacketIndex(kf.packetIndex);
        elementRoot.setPacketSize(kf.packetSize);
        elementRoot.setFrameSize(kf.frameSize);
        elementRoot.setFrameWidth(kf.width);
        elementRoot.setFrameHeight(kf.height);
        elementRoot.setQuality(kf.quality);
        elementRoot.setBitsPerPixel(kf.bitsPerPixel);
        elementRoot.setRawSize(kf.rawSize);
        keyFrameList.setWithCaveats(static_cast<capnp::uint>(i++), elementRoot);
    }
    root.setKeyFrames(keyFrameList);

    context.getResults().setVideoInfo(root);
    return kj::READY_NOW;
}

kj::Promise<void> VideoArchiveImpl::getVideoList(GetVideoListContext context) {
    auto collect_videos = [](const string& ps) {
        path p{ps};
        map<size_t, path> res;
        try {
            for (const auto& entry : recursive_directory_iterator(p))
                if (entry.is_regular_file()) {
                    const auto  fn{entry.path().filename().string()};
                    const regex video_extension_pattern{R"(.*\.(webm|mp4|mkv)(\.part)?$)"};
                    // const regex video_extension_pattern{R"(.*\.(mp4)(\.part)?$)"};
                    if (regex_match(fn, video_extension_pattern)) res.emplace(file_size(entry), entry.path());
                }
        }
        catch (const filesystem_error& e) {
            cerr << e.what() << endl;
        }
        return res;
    };
    auto dir = context.getParams().getFolderPath();
    const auto videos = collect_videos(dir);
    KJ_DBG("VideoArchiveImpl::getVideoList", dir, videos.size());
    auto   builder   = kj::heap<capnp::MallocMessageBuilder>();
    auto   root      = builder->initRoot<VideoList>();
    auto   videoList = root.initVideos(static_cast<unsigned int>(videos.size()));
    size_t i         = 0;
    for (const auto& [video_size, video_path] : videos) {
        auto elementBuilder = kj::heap<capnp::MallocMessageBuilder>();
        auto elementRoot    = elementBuilder->initRoot<Video>();
        elementRoot.setName(video_path.generic_string());
        elementRoot.setSizeBytes(static_cast<unsigned int>(video_size));
        videoList.setWithCaveats(static_cast<capnp::uint>(i++), elementRoot);
    }

    context.getResults().setVideoList(root.asReader());
    return kj::READY_NOW;
}
