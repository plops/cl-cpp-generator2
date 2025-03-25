//
// Created by martin on 3/25/25.
//

#include "FixedSizeImagePool.h"
FixedSizeImagePool::FixedSizeImagePool(size_t capacity, int width, int height) :
    width_{width}, height_{height}, capacity_{capacity}, images_{make_unique<GrayscaleImage[]>(capacity)},
    available_(capacity_, true) {}
FixedSizeImagePool::~FixedSizeImagePool() {}
IImage* FixedSizeImagePool::acquireImage() {
    unique_lock lock{mutex_};
    while (true) {
        for (size_t i = 0; i < capacity_; ++i) {
            if (available_[i]) {
                available_[i] = false;
                cv_.notify_one(); // Signal a waiting thread
                return &images_[i];
            }
        }
        cv_.wait(lock); // wait for an image to become available
    }
}
void    FixedSizeImagePool::releaseImage(IImage* image) {}
