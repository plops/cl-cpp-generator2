#ifndef SOKOLAPP_H
#define SOKOLAPP_H

#include <memory>
class sokolappImpl;
// placeholder public-header-preamble
;
class sokolapp  {
        public:
        
        std::unique_ptr<sokolappImpl> pimpl;
         sokolapp ()     ;  
         ~sokolapp ()     ;  
        // placeholder public-code-inside-class
;
};

#endif /* !SOKOLAPP_H */