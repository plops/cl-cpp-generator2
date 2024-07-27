#include <readerwriterqueue.h>

int main()
{
    moodycamel::ReaderWriterQueue<int> q(100);
}
