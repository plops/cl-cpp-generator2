- https://www.youtube.com/watch?v=yUIFdL3D0Vk
  - Contemporary C++ in Action - Daniela Engert - CppCon 2022
  - https://github.com/DanielaE/CppInAction
  - this is a windows project
  - c_resource.hpp is similar to 112_usb, but expands on the idea


- look at the function calls

```
[martin@localhost source00]$ ltrace -Cf ./main
[pid 2142168] std::ios_base::Init::Init()(0x404254, 0x7fff99673cd8, 0x7fff99673ce8, 0x403dc8) = 0x7feffd91e5f8
[pid 2142168] __cxa_atexit(0x401210, 0x404254, 0x402008, 6) = 0
[pid 2142168] SDL_Init(32, 0x7feffd918300, 0xd5859e05c223daa7, 288) = 0
[pid 2142168] std::ios_base::Init::Init()(0x40425c, 0x7fff99673cd8, 0x7fff99673ce8, 0x403dd0) = 2
[pid 2142168] __cxa_atexit(0x401210, 0x40425c, 0x402008, 0x403dd0) = 0
[pid 2142168] SDL_GetNumVideoDisplays(0x7fff99673b98, 0xf00140, 0x7fff99673ce8, 0x403dd8) = 1
[pid 2142168] SDL_GetDisplayBounds(0, 0x7fff99673b38, 0x7fff99673ce8, 0x403dd8) = 0
[pid 2142168] SDL_CreateWindow(0x402020, 800, 420, 320) = 0x1f95f70
[pid 2142168] std::basic_ostream<char, std::char_traits<char> >& std::__ostream_insert<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*, long)(0x404140, 0x40202c, 11, 0) = 0x404140
[pid 2142168] std::ctype<char>::_M_widen_init() const(0x7feffdc32700, 51, 0x7feffdc2b210, 1024) = 1
[pid 2142168] std::ostream::put(char)(0x404140, 10, 256, 0construct83
) = 0x404140
[pid 2142168] std::ostream::flush()(0x404140, 0x207cd60, 0x7feffdc2b210, 1) = 0x404140
[pid 2142168] std::basic_ostream<char, std::char_traits<char> >& std::__ostream_insert<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*, long)(0x404140, 0x402038, 9, 3072) = 0x404140
[pid 2142168] std::ostream::put(char)(0x404140, 10, 0x7feffdc2b210, 0x3d726f7461726570operator=
) = 0x404140
[pid 2142168] std::ostream::flush()(0x404140, 0x207cd60, 0x7feffdc2b210, 1) = 0x404140
[pid 2142168] std::basic_ostream<char, std::char_traits<char> >& std::__ostream_insert<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&, char const*, long)(0x404140, 0x402052, 11, 3072) = 0x404140
[pid 2142168] std::ostream::put(char)(0x404140, 10, 0x7feffdc2b210, 0x3932317463757274destruct129
) = 0x404140
[pid 2142168] std::ostream::flush()(0x404140, 0x207cd60, 0x7feffdc2b210, 1) = 0x404140



```