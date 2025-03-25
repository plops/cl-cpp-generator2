//
// Created by martin on 3/25/25.
//

#include "GrayscaleImage.h"
GrayscaleImage::GrayscaleImage(int width, int height) :
    width_(width), height_(height), data_{make_unique<uint8_t[]>(width * height)} {}

GrayscaleImage::~GrayscaleImage() {}
int      GrayscaleImage::getWidth() const { return width_; }
int      GrayscaleImage::getHeight() const { return height_; }
uint8_t* GrayscaleImage::getData() { return data_.get(); }
