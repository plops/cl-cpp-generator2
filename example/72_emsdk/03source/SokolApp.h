#ifndef SOKOLAPP_H
#define SOKOLAPP_H

#include <memory>
class SokolAppImpl;
// placeholder public-header-preamble
;
class SokolApp  {
        public:
        
        std::unique_ptr<SokolAppImpl> pimpl;
         SokolApp ()     ;  
         ~SokolApp ()     ;  
        // placeholder public-code-inside-class
;
};

#endif /* !SOKOLAPP_H */