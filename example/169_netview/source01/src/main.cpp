#include <filesystem>
#include <iostream>
using namespace std;
using namespace std::filesystem;
int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <path>" << endl;
        return 1;
    }
   path f{argv[1]};

}
