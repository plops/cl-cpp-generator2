#ifndef FFTWMANAGER_H
#define FFTWMANAGER_H

#include <fftw3.h>
#include <map>
#include <vector>
#include <complex> 

class FFTWManager  {
        public:
        explicit  FFTWManager ()       ;   
        fftw_plan get_plan (int windowSize)       ;   
        std::vector<std::complex<double>> fft (const std::vector<std::complex<double>>& in, int windowSize)       ;   
         ~FFTWManager ()       ;   
        private:
        std::map<int,fftw_plan> plans_;
};

#endif /* !FFTWMANAGER_H */