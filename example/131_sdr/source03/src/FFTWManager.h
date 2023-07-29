#ifndef FFTWMANAGER_H
#define FFTWMANAGER_H

#include <fftw3.h>
#include <map>
#include <vector>
#include <complex> 

class FFTWManager  {
        public:
        explicit  FFTWManager (int window_size)       ;   
         ~FFTWManager ()       ;   
        private:
        std::map<int,fftw_plan> plans_;
        int window_size_;
};

#endif /* !FFTWMANAGER_H */