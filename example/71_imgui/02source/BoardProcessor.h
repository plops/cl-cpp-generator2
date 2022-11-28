#pragma once
#include <mutex>
#include "MessageQueue.h"
#include "ProcessFrameEvent.h"
#include "ProcessedFrameMessage.h"
#include <vector>
#include <future>
class BoardProcessor  {
        public:
        bool run;
        int id;
        std::shared_ptr<MessageQueue<ProcessFrameEvent> > events;
        std::shared_ptr<MessageQueue<ProcessedFrameMessage> > msgs;
         BoardProcessor (int id_, std::shared_ptr<MessageQueue<ProcessFrameEvent> > events_, std::shared_ptr<MessageQueue<ProcessedFrameMessage> > msgs_)     ;  
        bool get_run ()     ;  
        int get_id ()     ;  
        std::shared_ptr<MessageQueue<ProcessFrameEvent> > get_events ()     ;  
        std::shared_ptr<MessageQueue<ProcessedFrameMessage> > get_msgs ()     ;  
        void process ()     ;  
        void processEvent (ProcessFrameEvent event)     ;  
        void stop ()     ;  
};
