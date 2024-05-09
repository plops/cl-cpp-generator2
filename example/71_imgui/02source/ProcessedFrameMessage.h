#pragma once
class ProcessedFrameMessage  {
        public:
        int batch_idx; 
        int frame_idx; 
        double seconds; 
         ProcessedFrameMessage (int batch_idx_, int frame_idx_, double seconds_)       ;   
        int get_batch_idx () const       ;   
        int get_frame_idx () const       ;   
        double get_seconds () const       ;   
};
