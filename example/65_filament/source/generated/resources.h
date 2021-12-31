#ifndef RESOURCES_H_
#define RESOURCES_H_

#include <stdint.h>

extern "C" {
    extern const uint8_t RESOURCES_PACKAGE[];
    extern int RESOURCES_BAKEDCOLOR_OFFSET;
    extern int RESOURCES_BAKEDCOLOR_SIZE;
}
#define RESOURCES_BAKEDCOLOR_DATA (RESOURCES_PACKAGE + RESOURCES_BAKEDCOLOR_OFFSET)

#endif
