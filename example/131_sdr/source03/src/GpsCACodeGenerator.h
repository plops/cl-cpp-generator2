#ifndef GPSCACODEGENERATOR_H
#define GPSCACODEGENERATOR_H

#include <vector>
#include <array>
#include <deque>
#include <cstddef> 

class GpsCACodeGenerator  {
        public:
        explicit  GpsCACodeGenerator (int prn)       ;   
        std::vector<bool> generate_sequence (size_t n)       ;   
        void print_square (const std::vector<bool> & v)       ;   
        private:
        bool step ()       ;   
        static constexpr size_t     register_size_=10;


        static constexpr std::array<int,2>     g1_feedback_bits_={3, 10};


        static constexpr std::array<int,6>     g2_feedback_bits_={2, 3, 6, 8, 9, 10};


        static constexpr std::array<std::pair<int,int>,32>     g2_shifts_={std::make_pair(2, 6), std::make_pair(3, 7), std::make_pair(4, 8), std::make_pair(5, 9), std::make_pair(1, 9), std::make_pair(2, 10), std::make_pair(1, 8), std::make_pair(2, 9), std::make_pair(3, 10), std::make_pair(2, 3), std::make_pair(3, 4), std::make_pair(5, 6), std::make_pair(6, 7), std::make_pair(7, 8), std::make_pair(8, 9), std::make_pair(9, 10), std::make_pair(1, 4), std::make_pair(2, 5), std::make_pair(3, 6), std::make_pair(4, 7), std::make_pair(5, 8), std::make_pair(6, 9), std::make_pair(1, 3), std::make_pair(4, 6), std::make_pair(5, 7), std::make_pair(6, 8), std::make_pair(7, 9), std::make_pair(8, 10), std::make_pair(1, 6), std::make_pair(2, 7), std::make_pair(3, 8), std::make_pair(4, 9)};


        std::deque<bool> g1_;
        std::deque<bool> g2_;
        int prn_;
};

#endif /* !GPSCACODEGENERATOR_H */