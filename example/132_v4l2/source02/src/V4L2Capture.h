#ifndef V4L2CAPTURE_H
#define V4L2CAPTURE_H

#include <functional>

class V4L2Capture {
public:
    explicit V4L2Capture(const std::string &device, int buffer_count);

    ~V4L2Capture();

    void startCapturing();

    void stopCapturing();

    void setupFormat(int width, int height, int pixelFormat);

    void getFrameAndStore(const std::string &filename);

    void getFrame(std::function<void(void *, size_t)> fun);

private:
    struct buffer {
        void *start;
        size_t length;
    };


    void xioctl(unsigned long request, void *arg, const std::string &str);

    const std::string &device_;
    int buffer_count_;
    std::vector<buffer> buffers_;
    int fd_;
    int width_;
    int height_;
};

#endif /* !V4L2CAPTURE_H */