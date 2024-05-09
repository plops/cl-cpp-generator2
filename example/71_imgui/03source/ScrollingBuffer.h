#pragma once
#include "imgui.h" 
class ScrollingBuffer  {
        public:
        int max_size; 
        int offset; 
        ImVector<ImVec2> data; 
         ScrollingBuffer (int max_size_ = 200)       ;   
        void AddPoint (float x, float y)       ;   
        void Erase ()       ;   
};
