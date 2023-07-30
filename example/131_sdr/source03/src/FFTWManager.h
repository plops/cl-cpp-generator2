#ifndef FFTWMANAGER_H
#define FFTWMANAGER_H

#include <fftw3.h>
#include <map>
#include <vector>
#include <complex> 

class FFTWManager  {
        public:
        explicit  FFTWManager ()       ;   
        void print_plans () const      ;   
        fftw_plan get_plan (int windowSize, int nThreads = 1)       ;   
        std::vector<std::complex<double>> fftshift (const std::vector<std::complex<double>>& in)       ;   
        std::vector<std::complex<double>> fft (const std::vector<std::complex<double>>& in, int windowSize)       ;   
         ~FFTWManager ()       ;   
        private:
        std::map<std::pair<int,int>,fftw_plan> plans_;
};

#endif /* !FFTWMANAGER_H */