// Stroustrup A Tour of C++ (2022) p. 151

#include <filesystem>
#include <iostream>
#include <regex>
#include <map>
using namespace std;
using namespace std::filesystem;
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <path>" << endl;
        return 1;
    }
    path p{argv[1]};
    auto collect_videos = [](const path& p)
    {
        map<size_t,path> res;
        try
        {
            if (is_directory(p))
            {
                for (const auto& entry : recursive_directory_iterator(p))
                {
                    if (entry.is_regular_file())
                    {
                        const auto fn{entry.path().filename().string()};
                        const regex video_extension_pattern{R"(.*\.(webm|mp4|mkv)(\.part)?$)"};
                        if (regex_match(fn, video_extension_pattern))
                        {
                            auto s{file_size(entry)};
                            res.emplace(s,entry.path());
                        }
                    }
                }
            }
        }
        catch (const filesystem_error& e)
        {
            cerr << e.what() << endl;
        }
        return res;
    };
    auto videos = collect_videos(p);
    for (const auto& [size, video_path] : videos)
        cout << size << " " << video_path << endl;
    return 0;
}
