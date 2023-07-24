// no preamble

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

#include "V4L2Capture.h"

V4L2Capture::V4L2Capture(const std::string &device, int buffer_count) : device_(device), buffer_count_(buffer_count),
                                                                        buffers_(0), fd_(0), width_(0), height_(0) {
    fd_ = open(device_.c_str(), O_RDWR);


    if (-1 == fd_) {
        throw std::runtime_error("opening video device failed" + std::string(std::strerror(errno)));

    }
}

V4L2Capture::~V4L2Capture() {
    for (const auto &b: buffers_) {
        munmap(b.start, b.length);
    }
    close(fd_);
}

void V4L2Capture::startCapturing() {
    std::cout << "startCapturing" << std::endl;
    auto type = v4l2_buf_type(V4L2_BUF_TYPE_VIDEO_CAPTURE);
    xioctl(VIDIOC_STREAMON, &type, "STREAMON");

}

void V4L2Capture::stopCapturing() {
    std::cout << "stopCapturing" << std::endl;
    auto type = v4l2_buf_type(V4L2_BUF_TYPE_VIDEO_CAPTURE);
    xioctl(VIDIOC_STREAMOFF, &type, "STREAMOFF");

}

void V4L2Capture::setupFormat(int width, int height, int pixelFormat) {
    std::cout << "setupFormat" << " width='" << width << "' " << " height='" << height << "' " << " pixelFormat='"
              << pixelFormat << "' " << std::endl;
    auto str = v4l2_streamparm({.type=V4L2_BUF_TYPE_VIDEO_CAPTURE});
    xioctl(VIDIOC_G_PARM, &str, "g-parm");
    str.parm.capture.timeperframe.numerator = 1;

    str.parm.capture.timeperframe.denominator = 10;

    xioctl(VIDIOC_S_PARM, &str, "s-parm");

    auto f = v4l2_format({.type=V4L2_BUF_TYPE_VIDEO_CAPTURE});
    f.fmt.pix.pixelformat = pixelFormat;
    f.fmt.pix.width = width;
    f.fmt.pix.height = height;
    f.fmt.pix.field = V4L2_FIELD_ANY;

    xioctl(VIDIOC_S_FMT, &f, "s-fmt");
    if (!(f.fmt.pix.pixelformat == pixelFormat)) {
        std::cout << "warning: we don't get the requested pixel format" << " f.fmt.pix.pixelformat='"
                  << f.fmt.pix.pixelformat << "' " << " pixelFormat='" << pixelFormat << "' " << std::endl;

    }
    width_ = f.fmt.pix.width;
    height_ = f.fmt.pix.height;

    auto r = v4l2_requestbuffers({.count=buffer_count_, .type=V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory=V4L2_MEMORY_MMAP});
    std::cout << "prepare several buffers" << " buffer_count_='" << buffer_count_ << "' " << std::endl;
    xioctl(VIDIOC_REQBUFS, &r, "reqbufs");
    buffers_.resize(r.count);
    for (auto i = 0; i < r.count; i += 1) {
        auto buf = v4l2_buffer({.index=i, .type=V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory=V4L2_MEMORY_MMAP});
        xioctl(VIDIOC_QUERYBUF, &buf, "querybuf");
        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);

        std::cout << "mmap memory for buffer" << " i='" << i << "' " << " buf.length='" << buf.length << "' "
                  << " (buffers_)[(i)].start='" << buffers_[i].start << "' " << std::endl;
        if (MAP_FAILED == buffers_[i].start) {
            throw std::runtime_error("mmap failed");

        }
        xioctl(VIDIOC_QBUF, &buf, "qbuf");

    }


}

void V4L2Capture::getFrameAndStore(const std::string &filename) {
    auto buf = v4l2_buffer({.type=V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory=V4L2_MEMORY_MMAP});
    xioctl(VIDIOC_DQBUF, &buf, "dqbuf");

    auto outFile = std::ofstream(filename, std::ios::binary);
    outFile << "P6\n" << width_ << " " << height_ << " 255\n";
    outFile.write(static_cast<char *>(buffers_[buf.index].start), buf.bytesused);
    outFile.close();
    xioctl(VIDIOC_QBUF, &buf, "qbuf");

}

void V4L2Capture::getFrame(std::function<void(void *, size_t)> fun) {
    auto buf = v4l2_buffer({.type=V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory=V4L2_MEMORY_MMAP});
    xioctl(VIDIOC_DQBUF, &buf, "dqbuf");

    auto b = buffers_[buf.index];
    fun(b.start, b.length);

    xioctl(VIDIOC_QBUF, &buf, "qbuf");
}

void V4L2Capture::xioctl(unsigned long request, void *arg, const std::string &str) {
    auto r = 0;
    do {
        r = ioctl(fd_, request, arg);


    } while (-1 == r && EINTR == errno);
    if (-1 == r) {
        throw std::runtime_error("ioctl " + str + " " + std::strerror(errno));

    }

} 
 
