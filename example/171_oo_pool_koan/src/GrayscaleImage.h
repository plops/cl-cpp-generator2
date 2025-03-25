//
// Created by martin on 3/25/25.
//

#ifndef GRAYSCALEIMAGE_H
#define GRAYSCALEIMAGE_H
#include "IImage.h"
#include <cstdint>
#include <memory>
using namespace std;
class GrayscaleImage : public IImage {
public:
    GrayscaleImage(int width, int height);
~GrayscaleImage() override;
    int getWidth() const override;
    int getHeight() const override;
    uint8_t* getData() override;
private:
    int width_;
    int height_;
    unique_ptr<uint8_t[]> data_;
};
#endif // GRAYSCALEIMAGE_H
