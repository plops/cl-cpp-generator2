#ifndef GPSCACODEGENERATOR_H
#define GPSCACODEGENERATOR_H

#include <vector>
#include <utility>
#include <array>
#include <cstddef> 

class GpsCACodeGenerator  {
        public:
        explicit  GpsCACodeGenerator (const int prn)       ;   
        std::vector<bool> generate_sequence (size_t n)       ;   
        private:
        int chip ()       ;   
        void clock ()       ;   
        bool step ()       ;   
        std::array<char,11>     g1_={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


        std::array<char,11>     g2_={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


        std::array<char*,2> tap_;
        static constexpr std::array<std::pair<int,int>,32>     lut_={std::make_pair(2, 6), std::make_pair(3, 7), std::make_pair(4, 8), std::make_pair(5, 9), std::make_pair(1, 9), std::make_pair(2, 10), std::make_pair(1, 8), std::make_pair(2, 9), std::make_pair(3, 10), std::make_pair(2, 3), std::make_pair(3, 4), std::make_pair(5, 6), std::make_pair(6, 7), std::make_pair(7, 8), std::make_pair(8, 9), std::make_pair(9, 10), std::make_pair(1, 4), std::make_pair(2, 5), std::make_pair(3, 6), std::make_pair(4, 7), std::make_pair(5, 8), std::make_pair(6, 9), std::make_pair(1, 3), std::make_pair(4, 6), std::make_pair(5, 7), std::make_pair(6, 8), std::make_pair(7, 9), std::make_pair(8, 10), std::make_pair(1, 6), std::make_pair(2, 7), std::make_pair(3, 8), std::make_pair(4, 9)};


        const int prn_;
};

#endif /* !GPSCACODEGENERATOR_H */