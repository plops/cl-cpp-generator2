#ifndef JXLENCODE_H
#define JXLENCODE_H

#include <jxl/encode_cxx.h>
#include <vector> 
class JxlEncode  {
        public:
        JxlThreadParallelRunnerPtr runner; 
         JxlEncode ()       ;   
        void Encode ()       ;   
         ~JxlEncode ()       ;   
};

#endif /* !JXLENCODE_H */