//
// Created by martin on 3/25/25.
//

#ifndef IIMAGEPOOL_H
#define IIMAGEPOOL_H

#include "IImage.h"

class IImagePool {
public:
    virtual ~IImagePool()                       = default;
    virtual IImage* acquireImage()              = 0;
    virtual void    releaseImage(IImage* image) = 0;
};


#endif // IIMAGEPOOL_H
