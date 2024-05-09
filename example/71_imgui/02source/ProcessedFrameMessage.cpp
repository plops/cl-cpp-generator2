// no preamble

#include "ProcessedFrameMessage.h"
ProcessedFrameMessage::ProcessedFrameMessage(int batch_idx_, int frame_idx_,
                                             double seconds_)
    : batch_idx{batch_idx_}, frame_idx{frame_idx_}, seconds{seconds_} {}
int ProcessedFrameMessage::get_batch_idx() const { return batch_idx; }
int ProcessedFrameMessage::get_frame_idx() const { return frame_idx; }
double ProcessedFrameMessage::get_seconds() const { return seconds; }