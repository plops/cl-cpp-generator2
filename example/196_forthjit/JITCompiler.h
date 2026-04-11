#pragma once
// header 
class JITCompiler  {
        public:
         ~JITCompiler ()       ;   
        private:
        context ctx {0};
        ForthVM& vm {0};
};
