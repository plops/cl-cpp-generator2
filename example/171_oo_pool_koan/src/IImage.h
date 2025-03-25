//
// Created by martin on 3/25/25.
//

#ifndef IIMAGE_H
#define IIMAGE_H
#include <cstdint>
class IImage {
public:
    virtual ~IImage()                   = default;
    virtual int      getWidth() const  = 0;
    virtual int      getHeight() const = 0;
    virtual uint8_t* getData()         = 0;
};
#endif // IIMAGE_H
