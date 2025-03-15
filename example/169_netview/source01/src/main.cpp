// Stroustrup A Tour of C++ (2022) p. 151

#include <filesystem>
#include <iostream>
#include <regex>
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
        try
        {
            if (is_directory(p))
            {
                for (const auto& entry : recursive_directory_iterator(p))
                {
                    if (entry.is_regular_file())
                    {
                        auto fn{entry.path().filename().string()};
                        regex pat{R"(.*\.(webm|mp4|mkv)(\.part)?$)"};
                        auto match = regex_match(fn, pat);
                        if (match)
                        {
                            cout << fn << endl;
                        }
                    }
                }
            }
        }
        catch (const filesystem_error& e)
        {
            cerr << e.what() << endl;
        }
    };
    collect_videos(p);
    return 0;
}
