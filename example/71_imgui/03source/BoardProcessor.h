#pragma once
#include <mutex>
#include "MessageQueue.h"
#include "ProcessFrameEvent.h"
#include "ProcessedFrameMessage.h"
#include "Charuco.h"
#include <opencv2/core/mat.hpp>
#include <vector>
#include <future> 
class BoardProcessor  {
        public:
        bool run; 
        int id; 
        std::shared_ptr<MessageQueue<ProcessFrameEvent> > events; 
        std::shared_ptr<MessageQueue<ProcessedFrameMessage> > msgs; 
        Charuco charuco; 
         BoardProcessor (int id_, std::shared_ptr<MessageQueue<ProcessFrameEvent> > events_, std::shared_ptr<MessageQueue<ProcessedFrameMessage> > msgs_, Charuco charuco_)       ;   
        bool get_run ()       ;   
        int get_id ()       ;   
        std::shared_ptr<MessageQueue<ProcessFrameEvent> > get_events ()       ;   
        std::shared_ptr<MessageQueue<ProcessedFrameMessage> > get_msgs ()       ;   
        Charuco get_charuco ()       ;   
        void process ()       ;   
        void processEvent (ProcessFrameEvent event)       ;   
        void stop ()       ;   
};
