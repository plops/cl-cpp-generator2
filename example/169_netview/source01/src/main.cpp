// Stroustrup A Tour of C++ (2022) p. 151

#include <filesystem>
#include <iostream>
#include <map>
#include <regex>
#include <format>
#include "proto/video.capnp.h"

#include <kj/debug.h>
#include <capnp/ez-rpc.h>
#include <capnp/message.h>

extern "C" {
// #include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

/*
 * Main components
 *
 * Format (Container) - a wrapper, providing sync, metadata and muxing for the streams.
 * Stream - a continuous stream (audio or video) of data over time.
 * Codec - defines how data are enCOded (from Frame to Packet)
 *         and DECoded (from Packet to Frame).
 * Packet - are the data (kind of slices of the stream data) to be decoded as raw frames.
 * Frame - a decoded raw frame (to be encoded or filtered).
 */

using namespace std;
using namespace std::filesystem;


auto collect_videos = [](const path& p)
{
    map<size_t, path> res;
    try
    {
        for (const auto& entry : recursive_directory_iterator(p))
            if (entry.is_regular_file())
            {
                const auto fn{entry.path().filename().string()};
                const regex video_extension_pattern{R"(.*\.(webm|mp4|mkv)(\.part)?$)"};
                if (regex_match(fn, video_extension_pattern))
                    res.emplace(file_size(entry), entry.path());
            }
    }
    catch (const filesystem_error& e)
    {
        cerr << e.what() << endl;
    }
    return res;
};

class VideoArchiveImpl final: public VideoArchive::Server
{
public:
    VideoArchiveImpl();
    kj::Promise<void> getVideoList(GetVideoListContext context) override;
};

VideoArchiveImpl::VideoArchiveImpl()
{

}

kj::Promise<void> VideoArchiveImpl::getVideoList(GetVideoListContext context)
{
    cout << "ServerImpl::getVideoList" << endl;

    const auto videos = collect_videos("/mnt5/tmp/bb");

    auto builder = kj::heap<capnp::MallocMessageBuilder>();
    auto root = builder->initRoot<VideoList>();
    auto videoList = root.initVideos(videos.size());
    size_t i = 0;
    for (const auto& [video_size, video_path] : videos)
    {
        auto elementBuilder = kj::heap<capnp::MallocMessageBuilder>();
        auto elementRoot = elementBuilder->initRoot<Video>();
        elementRoot.setName(video_path.generic_string());
        elementRoot.setSize(video_size);
        videoList.setWithCaveats(i++, elementRoot);
    }

    context.getResults().setVideoList(root.asReader());
    return kj::READY_NOW;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <path>" << endl;
        return 1;
    }
    path p{argv[1]};

    capnp::EzRpcServer server(kj::heap<VideoArchiveImpl>(),"*");

    const auto videos = collect_videos(p);
    path last;
    for (const auto& [size, video_path] : videos)
    {
        cout << size << " " << video_path.stem() << endl;
        last = video_path;
    }

    auto version = avformat_version();
    auto versionStr = format("libavformat: {}.{}.{}",
        AV_VERSION_MAJOR(version),AV_VERSION_MINOR(version),
        AV_VERSION_MICRO(version) );

    cout << versionStr << endl;

    auto ctx = avformat_alloc_context();
    if (!ctx)
    {
        cerr << "Could not allocate video context" << endl;
        return 1;
    }
    return 0;
}
