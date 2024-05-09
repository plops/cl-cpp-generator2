// no preamble

#include "ProcessFrameEvent.h"
ProcessFrameEvent::ProcessFrameEvent(int batch_idx_, int frame_idx_, int dim_,
                                     float fps_, double seconds_,
                                     cv::Mat frame_)
    : batch_idx{batch_idx_}, frame_idx{frame_idx_}, dim{dim_}, fps{fps_},
      seconds{seconds_}, frame{frame_} {}
int ProcessFrameEvent::get_batch_idx() { return batch_idx; }
int ProcessFrameEvent::get_frame_idx() { return frame_idx; }
int ProcessFrameEvent::get_dim() { return dim; }
float ProcessFrameEvent::get_fps() { return fps; }
double ProcessFrameEvent::get_seconds() { return seconds; }
cv::Mat ProcessFrameEvent::get_frame() { return frame; }