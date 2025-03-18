//
// Created by martin on 3/16/25.
//

#include "VideoArchiveImpl.h"

#include <capnp/message.h>

#include <filesystem>
#include <iostream>
#include <map>
#include <regex>
#include <kj/timer.h>
#include <capnp/rpc-prelude.h>

using namespace std;
using namespace std::filesystem;

VideoArchiveImpl::VideoArchiveImpl() = default;

kj::Promise<void> VideoArchiveImpl::getVideoList(GetVideoListContext context) {
    auto collect_videos = [](const path& p) {
        map<size_t, path> res;
        try {
            for (const auto& entry : recursive_directory_iterator(p))
                if (entry.is_regular_file()) {
                    const auto  fn{entry.path().filename().string()};
                    const regex video_extension_pattern{
                            R"(.*\.(webm|mp4|mkv)(\.part)?$)"};
                    if (regex_match(fn, video_extension_pattern))
                        res.emplace(file_size(entry), entry.path());
                }
        }
        catch (const filesystem_error& e) { cerr << e.what() << endl; }
        return res;
    };
    cout << "ServerImpl::getVideoList" << endl;

    const auto videos = collect_videos("/mnt5/tmp/bb");

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
