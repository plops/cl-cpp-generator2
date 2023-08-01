#ifndef FFTWMANAGER_H
#define FFTWMANAGER_H

#include <fftw3.h>
#include <map>
#include <vector>
#include <complex> 

class FFTWManager  {
        public:
        explicit  FFTWManager ()       ;   
        std::vector<std::complex<double>> fftshift (const std::vector<std::complex<double>>& in) const      ;   
        std::vector<std::complex<double>> fft (const std::vector<std::complex<double>>& in, int windowSize) const      ;   
         ~FFTWManager ()       ;   
        private:
        fftw_plan get_plan (int windowSize, int nThreads = 1) const      ;   
};

#endif /* !FFTWMANAGER_H */