#ifndef DIAGRAMBASE_H
#define DIAGRAMBASE_H

#include <vector>
#include <deque>
#include <string> 
struct DiagramData {
        std::string name; 
        std::deque<float> values; 
};
/** @brief The DiagramBase class represents a base class for diagrams.

*/ 
class DiagramBase  {
        public:
            /** @brief Constructs a DiagramBase object with the specified maximum number of cores, maximum number of points, and y-axis name.

@param max_cores The maximum number of cores.
@param max_points The maximum number of points.
@param name_y The name of the y-axis.

    */ 
        explicit  DiagramBase (int max_cores, int max_points, std::string name_y)       ;   
        void AddDataPoint (float time, const std::vector<float>& values)       ;   
        int GetMaxCores ()       ;   
        int GetMaxPoints ()       ;   
        std::vector<DiagramData> GetDiagrams ()       ;   
        std::string GetNameY ()       ;   
        std::deque<float> GetTimePoints ()       ;   
        int max_cores_;
        int max_points_;
        std::vector<DiagramData> diagrams_;
        std::string name_y_;
        std::deque<float> time_points_;
};

#endif /* !DIAGRAMBASE_H */