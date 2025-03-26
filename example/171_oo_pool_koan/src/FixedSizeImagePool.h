//
// Created by martin on 3/25/25.
//

#ifndef FIXEDSIZEIMAGEPOOL_H
#define FIXEDSIZEIMAGEPOOL_H
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
#include "GrayscaleImage.h"
#include "IImagePool.h"
class FixedSizeImagePool : public IImagePool {
public:
    FixedSizeImagePool(size_t capacity, int width, int height);
    ~FixedSizeImagePool();
    IImage* acquireImage() override;
    void    releaseImage(IImage* image) override;

private:
    const int                    width_;
    const int                    height_;
    const size_t                 capacity_;
    unique_ptr<GrayscaleImage[]> images_{nullptr};
    vector<bool>                 available_;
    mutex                        mutex_;
    condition_variable           cv_;
};


#endif // FIXEDSIZEIMAGEPOOL_H
