#ifndef DIAGRAMWITHGUI_H
#define DIAGRAMWITHGUI_H

#include <vector>
#include <deque>
#include <string> 
#include "DiagramBase.h" 
class DiagramWithGui : public DiagramBase {
        public:
        using DiagramBase::DiagramBase; 
        /**  * Renders the GUI for the DiagramWithGui class.
 * 
 * @param xticks A boolean value indicating whether to render x-axis ticks.
*/ 
        void RenderGui (bool xticks = false)       ;   
        /**  * Renders the GUI with summed diagrams. The sum is computed over all previous cpus.
 * 
 * @param xticks Flag indicating whether to render x-axis ticks.
*/ 
        void RenderGuiSum (bool xticks = false)       ;   
};

#endif /* !DIAGRAMWITHGUI_H */