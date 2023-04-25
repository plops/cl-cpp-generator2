#ifndef WAVETABLEOSCILLATOR_H
#define WAVETABLEOSCILLATOR_H

#include <vector>
#include <cstdint>

class WavetableOscillator  {
        public:
        explicit  WavetableOscillator (double sample_rate, std::vector<double> wavetable)       ;  
        void set_frequency (double frequency)       ;  
        double next_sample ()       ;  
        private:
        double sample_rate_;
        std::vector<double> wavetable_;
        std::size_t wavetable_size_;
        double current_index_;
        double step_;
};

#endif /* !WAVETABLEOSCILLATOR_H */