#ifndef FFTWMANAGER_H
#define FFTWMANAGER_H

#include <fftw3.h>
#include <map>
#include <vector>
#include <complex>
#include <cstddef>
#include <utility> 

class FFTWManager  {
        public:
        explicit  FFTWManager (int number_threads)       ;   
        [[nodiscard]] std::vector<std::complex<double>> fftshift (const std::vector<std::complex<double>>& in) const      ;   
        std::vector<std::complex<double>> fft (const std::vector<std::complex<double>>& in, size_t windowSize)       ;   
        std::vector<std::complex<double>> ifft (const std::vector<std::complex<double>>& in, size_t windowSize)       ;   
         ~FFTWManager ()       ;   
        private:
        fftw_plan get_plan (size_t windowSize, int direction = FFTW_FORWARD, int nThreads = 1)       ;   
        std::map<std::pair<int,int>,fftw_plan> plans_;
        int     number_threads_=6;


};

#endif /* !FFTWMANAGER_H */