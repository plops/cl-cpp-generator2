#ifndef DIAGRAMWITHGUI_H
#define DIAGRAMWITHGUI_H

#include <vector>
#include <deque>
#include <string> 
#include "DiagramBase.h" 
class DiagramWithGui : public DiagramBase {
        public:
        using DiagramBase::DiagramBase; 
        void RenderGui (bool xticks = false)       ;   
};

#endif /* !DIAGRAMWITHGUI_H */