#ifndef UTILS_H

#define UTILS_H

#define length(a) (sizeof((a)) / sizeof(*(a)))
#define max(a, b)                                                              \
  ({                                                                           \
    __auto_type _a = (a);                                                      \
    __auto_type _b = (b);                                                      \
    _a > _b ? _a : _b;                                                         \
  })
#define min(a, b)                                                              \
  ({                                                                           \
    __auto_type _a = (a);                                                      \
    __auto_type _b = (b);                                                      \
    _a < _b ? _a : _b;                                                         \
  })
;

#endif
