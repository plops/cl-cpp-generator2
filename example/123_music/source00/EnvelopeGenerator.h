#ifndef ENVELOPEGENERATOR_H
#define ENVELOPEGENERATOR_H

#include <vector>
#include <cstdint>

class EnvelopeGenerator  {
        public:
        explicit  EnvelopeGenerator (double sample_rate, double attack, double decay, double sustain, double release)       ;  
        void note_on ()       ;  
        void note_off ()       ;  
        double next_amplitude ()       ;  
        private:
        enum class EnvelopeGeneratorState {Idle, Attack, Decay, Sustain, Release};
        double sample_rate_;
        double attack_;
        double decay_;
        double sustain_;
        double release_;
        EnvelopeGeneratorState current_state_;
        double current_amplitude_;
        double attach_increment_;
        double decay_increment_;
        double release_increment_;
};

#endif /* !ENVELOPEGENERATOR_H */