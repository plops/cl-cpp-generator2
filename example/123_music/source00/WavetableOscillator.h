#ifndef WAVETABLEOSCILLATOR_H
#define WAVETABLEOSCILLATOR_H



class WavetableOscillator  {
        public:
        explicit  WavetableOscillator (double sample_rate, std::vector<double> wavetable)       ;  
        private:
        double sample_rate;
        std::vector<double> wavetable;
};

#endif /* !WAVETABLEOSCILLATOR_H */