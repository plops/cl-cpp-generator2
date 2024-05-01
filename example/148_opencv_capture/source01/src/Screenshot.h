#ifndef SCREENSHOT_H
#define SCREENSHOT_H


class Screenshot  {
        public:
        explicit  Screenshot (int x, int y, int width, int height)       ;   
        const std::unique_ptr<Display, decltype(&XCloseDisplay)>& GetDisplay () const      ;   
        const Window& GetRoot () const      ;   
        const XWindowAttributes& GetWindowAttributes () const      ;   
        const Screen*& GetScreen () const      ;   
        const int& GetX () const      ;   
        const int& GetY () const      ;   
        const int& GetWidth () const      ;   
        const int& GetHeight () const      ;   
        std::unique_ptr<Display, decltype(&XCloseDisplay)> display;
        Window root;
        XWindowAttributes window_attributes;
        Screen* screen;
        int x;
        int y;
        int width;
        int height;
};

#endif /* !SCREENSHOT_H */