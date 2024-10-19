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
        explicit  DiagramBase (unsigned long max_cores, unsigned int max_points, std::string name_y)       ;   
        /** @brief Adds a data point to the diagram.
 
  This function adds a data point to the diagram at the specified time with the given values.
  
  @param time The time of the data point.
  @param values The values of the data point.
*/ 
        void AddDataPoint (float time, const std::vector<float>& values)       ;   
        const unsigned long& GetMaxCores () const      ;   
        const unsigned int& GetMaxPoints () const      ;   
        const std::vector<DiagramData>& GetDiagrams () const      ;   
        const std::string& GetNameY () const      ;   
        const std::deque<float>& GetTimePoints () const      ;   
        unsigned long max_cores_;
        unsigned int max_points_;
        std::vector<DiagramData> diagrams_;
        std::string name_y_;
        std::deque<float> time_points_;
};

#endif /* !DIAGRAMBASE_H */