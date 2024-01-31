#ifndef DIAGRAMBASE_H
#define DIAGRAMBASE_H

#include <vector>
#include <deque>
#include <string> 
struct DiagramData {
        std::string name; 
        std::deque<float> values; 
};
class DiagramBase  {
        public:
        explicit  DiagramBase (int max_cores, int max_points, std::string name_y)       ;   
        void AddDataPoint (float time, const std::vector<float>& values)       ;   
        int GetMaxCores ()       ;   
        int GetMaxPoints ()       ;   
        std::vector<DiagramData> GetDiagrams ()       ;   
        std::string GetNameY ()       ;   
        std::deque<float> GetTimePoints ()       ;   
        protected:
        int max_cores_;
        int max_points_;
        std::vector<DiagramData> diagrams_;
        std::string name_y_;
        std::deque<float> time_points_;
};

#endif /* !DIAGRAMBASE_H */