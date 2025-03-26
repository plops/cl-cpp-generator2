//
// Created by martin on 3/25/25.
//

#ifndef GRAYSCALEIMAGE_H
#define GRAYSCALEIMAGE_H
#include <cstdint>
#include <memory>
#include "IImage.h"
using namespace std;
class GrayscaleImage : public IImage {
public:
    GrayscaleImage(int width = 128, int height = 64);
    ~GrayscaleImage() override;
    int      getWidth() const override;
    int      getHeight() const override;
    uint8_t* getData() override;

private:
    const int             width_;
    const int             height_;
    unique_ptr<uint8_t[]> data_{nullptr};
};
#endif // GRAYSCALEIMAGE_H
